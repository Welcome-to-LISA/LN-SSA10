import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.features import geometry_mask
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import os
from tqdm import tqdm
import fiona
from shapely.geometry import shape
import gc

# ======================= 配置参数 =======================
CONFIG = {
    'year': 2025,
    'msi_base_path': r'F:\MSI',
    'sar_base_path': r'F:\SAR',
    'msi_pattern': '{season}{year}_MSI.tif',
    'sar_pattern': '{season}{year}_SAR.tif',
    'train_label_path': r'F:\train.tif',
    'val_label_path': r'F:\val.tif',
    'shp_file': r'F:\potential_region.shp',
    'export_folder': r'F:\result',
    'export_prefix': 'LN_SSA{year}',
    'block_size': 2048,
    'label_read_block_size': 4096,
    'n_estimators': 100,
    'max_samples_per_class': 50000,
    'max_val_samples_per_class': 20000,
}

SEASONS = ['spring', 'summer', 'autumn', 'winter']
year = CONFIG['year']
os.makedirs(CONFIG['export_folder'], exist_ok=True)
os.environ['GDAL_CACHEMAX'] = '2048'
os.environ['GDAL_NUM_THREADS'] = 'ALL_CPUS'

msi_paths = {s: os.path.join(CONFIG['msi_base_path'], f'{s}{year}_MSI.tif') for s in SEASONS}
sar_paths = {s: os.path.join(CONFIG['sar_base_path'], f'{s}{year}_SAR.tif') for s in SEASONS}
export_prefix = CONFIG['export_prefix'].format(year=year)

# ==================== 1. 读取元数据 & 计算SAR窗口 ====================
print('读取影像元数据...')
ref_meta = None
sar_windows = {}

for season in SEASONS:
    with rasterio.open(msi_paths[season]) as src:
        if ref_meta is None:
            ref_meta = {'height': src.height, 'width': src.width,
                        'transform': src.transform, 'crs': src.crs, 'count': src.count}

    with rasterio.open(sar_paths[season]) as sar_src:
        msi_bounds = rasterio.transform.array_bounds(ref_meta['height'], ref_meta['width'], ref_meta['transform'])
        ul_row, ul_col = rasterio.transform.rowcol(sar_src.transform, msi_bounds[0], msi_bounds[3])
        lr_row, lr_col = rasterio.transform.rowcol(sar_src.transform, msi_bounds[2], msi_bounds[1])
        sar_windows[season] = Window(max(0, ul_col), max(0, ul_row), ref_meta['width'], ref_meta['height'])

print(f'MSI尺寸: {ref_meta["height"]}x{ref_meta["width"]}, 波段数: {ref_meta["count"]}')

# ==================== 2. 特征名称 ====================
base_msi_names = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12', 'SSSI', 'CRVI', 'NDTI', 'REDI']
msi_bands_count = ref_meta['count']
msi_name_list = base_msi_names[:msi_bands_count]
for i in range(len(msi_name_list), msi_bands_count):
    msi_name_list.append(f'B{i + 1}')

feature_names = [f'{s}_{b}' for s in SEASONS for b in msi_name_list]
feature_names += [f'{s}_VV' for s in SEASONS] + [f'{s}_VH' for s in SEASONS]
print(f'特征总数: {len(feature_names)}')

# ==================== 3. 读取Shapefile掩膜 ====================
print('读取Shapefile...')
with fiona.open(CONFIG['shp_file']) as shp:
    geometries = [shape(f['geometry']) for f in shp]
    shp_crs = rasterio.crs.CRS(shp.crs)
    ref_crs = rasterio.crs.CRS(ref_meta['crs'])

if shp_crs != ref_crs:
    from pyproj import Transformer
    from shapely.ops import transform as shapely_transform

    transformer = Transformer.from_crs(shp_crs, ref_crs, always_xy=True)
    geometries = [shapely_transform(transformer.transform, g) for g in geometries]

shp_mask = geometry_mask(geometries, out_shape=(ref_meta['height'], ref_meta['width']),
                         transform=ref_meta['transform'], invert=True).astype(np.uint8)
print(f'掩膜内像素: {shp_mask.sum():,}')


# ==================== 4. 分块读取标签 ====================
def read_label(path, block_size=4096):
    with rasterio.open(path) as src:
        arr = np.zeros((src.height, src.width), dtype=src.dtypes[0])
        for r in range(0, src.height, block_size):
            for c in range(0, src.width, block_size):
                w = Window(c, r, min(block_size, src.width - c), min(block_size, src.height - r))
                arr[r:r + w.height, c:c + w.width] = src.read(1, window=w)
        return arr, src.transform


print('读取标签...')
train_gt, train_tfm = read_label(CONFIG['train_label_path'], CONFIG['label_read_block_size'])
val_gt, val_tfm = read_label(CONFIG['val_label_path'], CONFIG['label_read_block_size'])
train_unique = np.unique(train_gt[train_gt > 0])
val_unique = np.unique(val_gt[val_gt > 0])
print(f'训练类别: {train_unique}, 验证类别: {val_unique}')


# ==================== 5. 提取样本 ====================
def extract_samples(gt, transform, max_per_class, src_dict, desc=''):
    samples = []
    classes = np.unique(gt[gt > 0])
    for cls in classes:
        rows, cols = np.where(gt == cls)
        n = min(len(rows), max_per_class)
        idx = np.random.choice(len(rows), n, replace=False)
        for r, c in tqdm(zip(rows[idx], cols[idx]), total=n, desc=desc):
            x, y = rasterio.transform.xy(transform, r, c)
            img_r, img_c = rasterio.transform.rowcol(ref_meta['transform'], x, y)
            if not (0 <= img_r < ref_meta['height'] and 0 <= img_c < ref_meta['width']):
                continue
            feats = []
            valid = True
            for s in SEASONS:
                m = src_dict[s]['msi'].read(window=Window(img_c, img_r, 1, 1))[:, 0, 0]
                sw = sar_windows[s]
                sar = src_dict[s]['sar'].read(window=Window(sw.col_off + img_c, sw.row_off + img_r, 1, 1))[:, 0, 0]
                if np.any(np.isnan(m)) or np.any(np.isinf(m)) or np.any(m == 0):
                    valid = False;
                    break
                if np.any(np.isnan(sar)) or np.any(np.isinf(sar)) or np.any(sar == 0):
                    valid = False;
                    break
                feats.extend(m.tolist() + sar.tolist())
            if valid:
                samples.append(feats + [cls])
    return pd.DataFrame(samples, columns=feature_names + ['class'])


print('提取训练样本...')
src_dict = {s: {'msi': rasterio.open(msi_paths[s]), 'sar': rasterio.open(sar_paths[s])} for s in SEASONS}
train_df = extract_samples(train_gt, train_tfm, CONFIG['max_samples_per_class'], src_dict, desc='训练')
del train_gt

print('提取验证样本...')
val_df = extract_samples(val_gt, val_tfm, CONFIG['max_val_samples_per_class'], src_dict, desc='验证')
del val_gt

for s in SEASONS:
    src_dict[s]['msi'].close();
    src_dict[s]['sar'].close()
gc.collect()

print(f'训练样本: {len(train_df)}, 验证样本: {len(val_df)}')

# ==================== 6. 训练模型 ====================
print('训练随机森林...')
X_train, y_train = train_df[feature_names].values, train_df['class'].values
X_val, y_val = val_df[feature_names].values, val_df['class'].values

clf = RandomForestClassifier(n_estimators=CONFIG['n_estimators'], min_samples_leaf=10,
                             max_depth=30, random_state=42, n_jobs=-1, verbose=1)
clf.fit(X_train, y_train)

# ==================== 7. 评估 ====================
from sklearn.metrics import confusion_matrix
y_val_pred = clf.predict(X_val)
print(f'训练精度: {accuracy_score(y_train, clf.predict(X_train)):.4f}')
print(f'验证精度: {accuracy_score(y_val, y_val_pred):.4f}')
print(classification_report(y_val, y_val_pred))
print('混淆矩阵:')
print(confusion_matrix(y_val, y_val_pred))
feat_imp = pd.DataFrame({'feature': feature_names, 'importance': clf.feature_importances_}).sort_values('importance',
                                                                                                        ascending=False)
feat_imp.to_csv(os.path.join(CONFIG['export_folder'], f'{export_prefix}_feature_importance.csv'), index=False)
print(feat_imp.head(20))

# ==================== 8. 分块预测 ====================
print('分块预测...')
meta_out = {'count': 1, 'dtype': 'uint8', 'nodata': 0, 'compress': 'lzw', 'tiled': True,
            'blockxsize': 256, 'blockysize': 256, 'crs': ref_meta['crs'], 'transform': ref_meta['transform'],
            'height': ref_meta['height'], 'width': ref_meta['width']}
meta_conf = {**meta_out, 'dtype': 'float32', 'nodata': -9999}

out_paths = {
    'all': os.path.join(CONFIG['export_folder'], f'{export_prefix}.tif'),
    'conf': os.path.join(CONFIG['export_folder'], f'{export_prefix}_confidence.tif'),
}
for c in range(1, 6):
    out_paths[f'c{c}'] = os.path.join(CONFIG['export_folder'], f'{export_prefix}_class{c}_only.tif')

src_dict = {s: {'msi': rasterio.open(msi_paths[s]), 'sar': rasterio.open(sar_paths[s])} for s in SEASONS}
h, w = ref_meta['height'], ref_meta['width']
bs = CONFIG['block_size']

with rasterio.open(out_paths['all'], 'w', **meta_out) as dst_all, \
        rasterio.open(out_paths['conf'], 'w', **meta_conf) as dst_conf:
    dst_cls = {c: rasterio.open(out_paths[f'c{c}'], 'w', **meta_out) for c in range(1, 6)}

    for row in tqdm(range(0, h, bs), desc='行块'):
        for col in range(0, w, bs):
            r_end, c_end = min(row + bs, h), min(col + bs, w)
            rsz, csz = r_end - row, c_end - col
            win = Window(col, row, csz, rsz)

            mask = shp_mask[row:r_end, col:c_end]
            if not mask.any():
                for dst in [dst_all, dst_conf] + list(dst_cls.values()):
                    dst.write(np.zeros((rsz, csz), dtype=np.uint8), 1, window=win)
                continue

            rows_v, cols_v = np.where(mask)
            n_valid = len(rows_v)

            # ====== 优化点1：批量读取整块影像 ======
            # 一次性读取当前块的所有波段数据，而不是逐像素读
            msi_blocks = {}
            sar_blocks = {}
            for s in SEASONS:
                # 读取MSI整块 (所有波段)
                msi_blocks[s] = src_dict[s]['msi'].read(window=win)  # shape: (bands, rsz, csz)
                # 读取SAR整块
                sw = sar_windows[s]
                sar_win = Window(sw.col_off + col, sw.row_off + row, csz, rsz)
                sar_blocks[s] = src_dict[s]['sar'].read(window=sar_win)  # shape: (2, rsz, csz)

            # ====== 优化点2：矢量化构建特征矩阵 ======
            # 预分配特征矩阵 (n_valid, n_features)
            n_msi_bands = msi_blocks[SEASONS[0]].shape[0]
            n_sar_bands = sar_blocks[SEASONS[0]].shape[0]
            feats_per_season = n_msi_bands + n_sar_bands
            n_features = len(SEASONS) * feats_per_season
            feats_arr = np.zeros((n_valid, n_features), dtype=np.float32)

            for si, s in enumerate(SEASONS):
                start_idx = si * feats_per_season
                # MSI波段：从整块中提取对应像素
                for bi in range(n_msi_bands):
                    feats_arr[:, start_idx + bi] = msi_blocks[s][bi, rows_v, cols_v]
                # SAR波段
                for bi in range(n_sar_bands):
                    feats_arr[:, start_idx + n_msi_bands + bi] = sar_blocks[s][bi, rows_v, cols_v]

            # ====== 优化点3：矢量化无效值检测 ======
            invalid_mask = (
                np.any(np.isnan(feats_arr), axis=1) |
                np.any(np.isinf(feats_arr), axis=1) |
                np.any(feats_arr == 0, axis=1)
            )

            # ====== 优化点4：只对有效像素预测 ======
            valid_idx = np.where(~invalid_mask)[0]
            cls_map = np.zeros((rsz, csz), dtype=np.uint8)
            conf_map = np.full((rsz, csz), -9999, np.float32)

            if len(valid_idx) > 0:
                valid_feats = feats_arr[valid_idx]
                preds = clf.predict(valid_feats)
                confs = np.max(clf.predict_proba(valid_feats), axis=1)

                # 回填到结果图
                r_valid = rows_v[valid_idx]
                c_valid = cols_v[valid_idx]
                cls_map[r_valid, c_valid] = preds
                conf_map[r_valid, c_valid] = confs

            # 写入输出文件
            dst_all.write(cls_map, 1, window=win)
            dst_conf.write(conf_map, 1, window=win)
            for cc in range(1, 6):
                dst_cls[cc].write((cls_map == cc).astype(np.uint8), 1, window=win)

    for dst in dst_cls.values():
        dst.close()

for s in SEASONS:
    src_dict[s]['msi'].close()
    src_dict[s]['sar'].close()

print('完成')