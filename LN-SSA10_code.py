import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.features import geometry_mask
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pandas as pd
import os
from tqdm import tqdm
import fiona
from shapely.geometry import shape
import gc

# Configuration
CONFIG = {
    'year': 2025,
    'msi_base_path': r'F:\MSI',
    'sar_base_path': r'F:\SAR',
    'msi_pattern': '{season}{year}_MSI.tif',
    'sar_pattern': '{season}{year}_SAR.tif',

    'train_label_path': r'F:\train.tif',
    'test_label_path': r'F:\test.tif',

    'shp_file': r'F:\Mask_LN-SSA10.shp',
    'export_folder': r'F:\result',

    'block_size': 2048,
    'label_read_block_size': 4096,

    'n_estimators': 100,
    'max_depth': 30,
    'min_samples_leaf': 10,
    'max_features': 'sqrt',
    'random_state': 42,

    'max_samples_per_class': 50000,
    'max_test_samples_per_class': 20000,

    'ssa_class_value': 3,

    'nodata_value': np.nan,
}

SEASONS = ['spring', 'summer', 'autumn', 'winter']
SAR_BAND_NAMES = ['VV', 'VH']

year = CONFIG['year']
os.makedirs(CONFIG['export_folder'], exist_ok=True)
os.environ['GDAL_CACHEMAX'] = '2048'
os.environ['GDAL_NUM_THREADS'] = 'ALL_CPUS'

msi_paths = {
    s: os.path.join(
        CONFIG['msi_base_path'],
        CONFIG['msi_pattern'].format(season=s, year=year)
    )
    for s in SEASONS
}
sar_paths = {
    s: os.path.join(
        CONFIG['sar_base_path'],
        CONFIG['sar_pattern'].format(season=s, year=year)
    )
    for s in SEASONS
}

export_prefix = f'{year}_LN-SSA10'

print('读取影像元数据...')
ref_meta = None
sar_windows = {}

for season in SEASONS:
    with rasterio.open(msi_paths[season]) as src:
        if ref_meta is None:
            ref_meta = {
                'height': src.height,
                'width': src.width,
                'transform': src.transform,
                'crs': src.crs,
                'count': src.count,
            }
        else:
            if (
                src.height != ref_meta['height'] or
                src.width != ref_meta['width'] or
                src.transform != ref_meta['transform'] or
                src.crs != ref_meta['crs']
            ):
                raise ValueError(f'{season} MSI 与参考MSI网格不一致，请先完成对齐。')

    with rasterio.open(sar_paths[season]) as sar_src:
        if sar_src.count != 2:
            raise ValueError(
                f'{season} SAR 波段数为 {sar_src.count}，代码要求两个波段并按 VV、VH 顺序存储。'
            )

        msi_bounds = rasterio.transform.array_bounds(
            ref_meta['height'], ref_meta['width'], ref_meta['transform']
        )
        ul_row, ul_col = rasterio.transform.rowcol(
            sar_src.transform, msi_bounds[0], msi_bounds[3]
        )
        lr_row, lr_col = rasterio.transform.rowcol(
            sar_src.transform, msi_bounds[2], msi_bounds[1]
        )

        sar_windows[season] = Window(
            max(0, ul_col),
            max(0, ul_row),
            ref_meta['width'],
            ref_meta['height'],
        )

print(
    f'MSI尺寸: {ref_meta["height"]}x{ref_meta["width"]}, '
    f'波段数: {ref_meta["count"]}'
)

# Feature names must match the feature stacking order
base_msi_names = [
    'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12',
    'SSSI', 'CRVI', 'NDTI', 'REDI'
]

msi_bands_count = ref_meta['count']
msi_name_list = base_msi_names[:msi_bands_count]
for i in range(len(msi_name_list), msi_bands_count):
    msi_name_list.append(f'Band{i + 1}')

feature_names = []
for season in SEASONS:
    feature_names.extend([f'{season}_{b}' for b in msi_name_list])
    feature_names.extend([f'{season}_{b}' for b in SAR_BAND_NAMES])

print(f'特征总数: {len(feature_names)}')

print('读取Shapefile...')
# Potential-distribution mask
with fiona.open(CONFIG['shp_file']) as shp:
    geometries = [shape(f['geometry']) for f in shp]
    shp_crs = rasterio.crs.CRS(shp.crs)
    ref_crs = rasterio.crs.CRS(ref_meta['crs'])

if shp_crs != ref_crs:
    from pyproj import Transformer
    from shapely.ops import transform as shapely_transform

    transformer = Transformer.from_crs(shp_crs, ref_crs, always_xy=True)
    geometries = [
        shapely_transform(transformer.transform, g) for g in geometries
    ]

shp_mask = geometry_mask(
    geometries,
    out_shape=(ref_meta['height'], ref_meta['width']),
    transform=ref_meta['transform'],
    invert=True,
).astype(bool)

print(f'掩膜内像素: {shp_mask.sum():,}')

def read_label(path, block_size=4096):
    with rasterio.open(path) as src:
        arr = np.zeros((src.height, src.width), dtype=src.dtypes[0])
        tfm = src.transform
        for r in range(0, src.height, block_size):
            for c in range(0, src.width, block_size):
                w = Window(
                    c,
                    r,
                    min(block_size, src.width - c),
                    min(block_size, src.height - r),
                )
                arr[
                    r:r + int(w.height),
                    c:c + int(w.width)
                ] = src.read(1, window=w)
        return arr, tfm

print('读取标签...')
train_gt, train_tfm = read_label(
    CONFIG['train_label_path'], CONFIG['label_read_block_size']
)
test_gt, test_tfm = read_label(
    CONFIG['test_label_path'], CONFIG['label_read_block_size']
)

train_unique = np.unique(train_gt[train_gt > 0])
test_unique = np.unique(test_gt[test_gt > 0])
print(f'训练类别: {train_unique}, 测试类别: {test_unique}')

# Sample extraction
def extract_samples(
    gt,
    transform,
    max_per_class,
    src_dict,
    rng,
    desc='',
):
    samples = []
    classes = np.unique(gt[gt > 0])

    for cls in classes:
        rows, cols = np.where(gt == cls)
        n = min(len(rows), max_per_class)

        if n == len(rows):
            idx = np.arange(len(rows))
        else:
            idx = rng.choice(len(rows), n, replace=False)

        for r, c in tqdm(
            zip(rows[idx], cols[idx]),
            total=n,
            desc=desc,
        ):
            x, y = rasterio.transform.xy(transform, r, c)
            img_r, img_c = rasterio.transform.rowcol(
                ref_meta['transform'], x, y
            )

            if not (
                0 <= img_r < ref_meta['height'] and
                0 <= img_c < ref_meta['width']
            ):
                continue

            feats = []
            valid = True

            for season in SEASONS:
                msi = src_dict[season]['msi'].read(
                    window=Window(img_c, img_r, 1, 1)
                )[:, 0, 0]

                sw = sar_windows[season]
                sar = src_dict[season]['sar'].read(
                    window=Window(
                        sw.col_off + img_c,
                        sw.row_off + img_r,
                        1,
                        1,
                    )
                )[:, 0, 0]

                if (
                    np.any(np.isnan(msi)) or
                    np.any(np.isinf(msi)) or
                    np.any(msi == 0)
                ):
                    valid = False
                    break

                if (
                    np.any(np.isnan(sar)) or
                    np.any(np.isinf(sar)) or
                    np.any(sar == 0)
                ):
                    valid = False
                    break

                feats.extend(msi.tolist())
                feats.extend(sar.tolist())

            if valid:
                samples.append(feats + [cls])

    return pd.DataFrame(samples, columns=feature_names + ['class'])

rng = np.random.default_rng(CONFIG['random_state'])

print('提取训练样本...')
src_dict = {
    s: {
        'msi': rasterio.open(msi_paths[s]),
        'sar': rasterio.open(sar_paths[s]),
    }
    for s in SEASONS
}

train_df = extract_samples(
    train_gt,
    train_tfm,
    CONFIG['max_samples_per_class'],
    src_dict,
    rng,
    desc='训练',
)
del train_gt

print('提取测试样本...')
test_df = extract_samples(
    test_gt,
    test_tfm,
    CONFIG['max_test_samples_per_class'],
    src_dict,
    rng,
    desc='测试',
)
del test_gt

for s in SEASONS:
    src_dict[s]['msi'].close()
    src_dict[s]['sar'].close()

gc.collect()

print(f'训练样本: {len(train_df)}, 测试样本: {len(test_df)}')

if train_df.empty:
    raise ValueError('训练样本为空，请检查标签和输入特征。')
if test_df.empty:
    raise ValueError('测试样本为空，请检查标签和输入特征。')

# RF training and test evaluation
print('训练随机森林...')
X_train = train_df[feature_names].values
y_train = train_df['class'].values
X_test = test_df[feature_names].values
y_test = test_df['class'].values

clf = RandomForestClassifier(
    n_estimators=CONFIG['n_estimators'],
    max_depth=CONFIG['max_depth'],
    min_samples_leaf=CONFIG['min_samples_leaf'],
    max_features=CONFIG['max_features'],
    random_state=CONFIG['random_state'],
    n_jobs=-1,
    verbose=1,
)
clf.fit(X_train, y_train)

if CONFIG['ssa_class_value'] not in clf.classes_:
    raise ValueError(
        f"ssa_class_value={CONFIG['ssa_class_value']} 不在训练类别 "
        f"{clf.classes_.tolist()} 中，请检查SSA的GT类别编码。"
    )

y_test_pred = clf.predict(X_test)

print(f'训练精度: {accuracy_score(y_train, clf.predict(X_train)):.4f}')
print(f'测试精度: {accuracy_score(y_test, y_test_pred):.4f}')
print(classification_report(y_test, y_test_pred, zero_division=0))
print('混淆矩阵:')
print(confusion_matrix(y_test, y_test_pred))

feat_imp = pd.DataFrame({
    'feature': feature_names,
    'importance': clf.feature_importances_,
}).sort_values('importance', ascending=False)

feat_imp_path = os.path.join(
    CONFIG['export_folder'],
    f'{export_prefix}_feature_importance.csv',
)
feat_imp.to_csv(feat_imp_path, index=False)
print(feat_imp.head(20))

# Block prediction
print('分块预测...')

# Output: S. salsa=1, other classes=0 inside mask; outside mask=NaN
meta_binary = {
    'driver': 'GTiff',
    'count': 1,
    'dtype': 'float32',
    'nodata': CONFIG['nodata_value'],
    'compress': 'lzw',
    'tiled': True,
    'blockxsize': 256,
    'blockysize': 256,
    'BIGTIFF': 'IF_SAFER',
    'crs': ref_meta['crs'],
    'transform': ref_meta['transform'],
    'height': ref_meta['height'],
    'width': ref_meta['width'],
}
meta_conf = meta_binary.copy()

out_paths = {
    'binary': os.path.join(
        CONFIG['export_folder'],
        f'{year}_LN-SSA10.tif',
    ),
    'conf': os.path.join(
        CONFIG['export_folder'],
        f'{year}_LN-SSA10-Prob.tif',
    ),
}

src_dict = {
    s: {
        'msi': rasterio.open(msi_paths[s]),
        'sar': rasterio.open(sar_paths[s]),
    }
    for s in SEASONS
}

h, w = ref_meta['height'], ref_meta['width']
bs = CONFIG['block_size']

try:
    with rasterio.open(out_paths['binary'], 'w', **meta_binary) as dst_binary, \
            rasterio.open(out_paths['conf'], 'w', **meta_conf) as dst_conf:

        for row in tqdm(range(0, h, bs), desc='行块'):
            for col in range(0, w, bs):
                r_end = min(row + bs, h)
                c_end = min(col + bs, w)
                rsz = r_end - row
                csz = c_end - col
                win = Window(col, row, csz, rsz)

                mask = shp_mask[row:r_end, col:c_end]

                if not mask.any():
                    nan_block = np.full(
                        (rsz, csz), np.nan, dtype=np.float32
                    )
                    dst_binary.write(nan_block, 1, window=win)
                    dst_conf.write(nan_block, 1, window=win)
                    continue

                rows_v, cols_v = np.where(mask)
                n_mask_pixels = len(rows_v)

                msi_blocks = {}
                sar_blocks = {}

                for season in SEASONS:
                    msi_blocks[season] = src_dict[season]['msi'].read(
                        window=win
                    )

                    sw = sar_windows[season]
                    sar_win = Window(
                        sw.col_off + col,
                        sw.row_off + row,
                        csz,
                        rsz,
                    )
                    sar_blocks[season] = src_dict[season]['sar'].read(
                        window=sar_win
                    )

                n_msi_bands = msi_blocks[SEASONS[0]].shape[0]
                n_sar_bands = sar_blocks[SEASONS[0]].shape[0]
                feats_per_season = n_msi_bands + n_sar_bands
                n_features = len(SEASONS) * feats_per_season

                if n_features != len(feature_names):
                    raise ValueError(
                        f'预测特征数({n_features})与训练特征数'
                        f'({len(feature_names)})不一致。'
                    )

                feats_arr = np.empty(
                    (n_mask_pixels, n_features), dtype=np.float32
                )

                for si, season in enumerate(SEASONS):
                    start_idx = si * feats_per_season

                    for bi in range(n_msi_bands):
                        feats_arr[:, start_idx + bi] = (
                            msi_blocks[season][bi, rows_v, cols_v]
                        )

                    for bi in range(n_sar_bands):
                        feats_arr[:, start_idx + n_msi_bands + bi] = (
                            sar_blocks[season][bi, rows_v, cols_v]
                        )

                # All pixels inside the mask must have valid input features
                invalid_mask = (
                    np.any(np.isnan(feats_arr), axis=1) |
                    np.any(np.isinf(feats_arr), axis=1) |
                    np.any(feats_arr == 0, axis=1)
                )

                if np.any(invalid_mask):
                    invalid_count = int(invalid_mask.sum())
                    raise ValueError(
                        f'影像块 row={row}, col={col} 的潜在分布区内发现 '
                        f'{invalid_count} 个包含 NaN、Inf 或 0 的像元。'
                        '为保证mask内每个像元都有真实的RF分类置信度，'
                        '请先补全/修正输入特征影像后再运行。'
                    )

                preds = clf.predict(feats_arr)
                probas = clf.predict_proba(feats_arr)
                confs = np.max(probas, axis=1).astype(np.float32)

                binary_preds = (
                    preds == CONFIG['ssa_class_value']
                ).astype(np.float32)

                binary_map = np.full(
                    (rsz, csz), np.nan, dtype=np.float32
                )
                conf_map = np.full(
                    (rsz, csz), np.nan, dtype=np.float32
                )

                binary_map[rows_v, cols_v] = binary_preds
                conf_map[rows_v, cols_v] = confs

                dst_binary.write(binary_map, 1, window=win)
                dst_conf.write(conf_map, 1, window=win)

finally:
    for s in SEASONS:
        src_dict[s]['msi'].close()
        src_dict[s]['sar'].close()

print('完成')
print(f'年度二值分类结果: {out_paths["binary"]}')
print(f'年度分类置信度结果: {out_paths["conf"]}')
print(f'特征重要性结果: {feat_imp_path}')
