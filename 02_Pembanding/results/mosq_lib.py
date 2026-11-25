# [lib] impor-dan-konstanta
import os, re, sys, math, json, time, glob, random, shutil, platform
from types import SimpleNamespace

import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf
import keras
from keras import layers as L
from keras import ops

SR = 8000            # sampling rate (Hz)
TARGET = 2400        # 1 epoch = 300 ms
GAINS = (1, 1.5, 2)  # faktor volume wingbeat G pada validation/test
CLASSES = ['Ae.Aegypti_F', 'Ae.Aegypti_M', 'Ae.Albopictus_F', 'Ae.Albopictus_M',
           'An.Dirus_F', 'An.Dirus_M', 'Cx.Quin_F', 'Cx.Quin_M']   # urutan sama dengan LabelEncoder
SHORT = ['Ae.aeg F', 'Ae.aeg M', 'Ae.alb F', 'Ae.alb M', 'An.dir F', 'An.dir M', 'Cx.quin F', 'Cx.quin M']

# [lib] segmentasi
_numbers = re.compile(r'(\d+)')


def numerical_sort(value):
    """Urutan file 'natural' (cut2 < cut10), sama dengan numericalSort di notebook rilis."""
    parts = _numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def group_key(row):
    """Kunci kelompok rekaman: spesies_sex_umur[_LowMic] (recreate_mosquito_filename_without_noise)."""
    key = (row[0] + '_' + row[1]).strip() + '_' + row[2].strip()
    return key + '_LowMic' if row[5].strip() == 'LowMic' else key


def load_epochs(data_dir):
    """Potong setiap wav menjadi epoch 300 ms dengan hop 150 ms + jitter 0..119 sampel.
    Mengembalikan list (group_key, label, is_lowmic, epoch[2400]) dalam urutan notebook rilis."""
    out = []
    for p in sorted(os.listdir(data_dir), key=numerical_sort):
        if not p.endswith('.wav'):
            continue
        row = p.split('_')
        if row[0] == 'An.Minimus':
            continue
        x, sr = sf.read(os.path.join(data_dir, p), dtype='float32')
        assert sr == SR, f'{p}: diharapkan 8 kHz, didapat {sr}'
        label, key, low = row[0] + '_' + row[1][1], group_key(row), row[5] == 'LowMic'
        for i in range(0, len(x), TARGET // 2):
            np.random.seed(i)                       # jitter deterministik seperti notebook rilis
            shift = np.random.randint(120)
            y = x[i + shift: i + TARGET + shift]
            if len(y) == TARGET:
                out.append((key, label, low, y))
    return out


def count_per_group(epochs):
    """Jumlah epoch per kelompok rekaman (count_data)."""
    counts = {}
    for key, *_ in epochs:
        counts[key] = counts.get(key, 0) + 1
    return counts


def epoch_files(data_dir):
    """Nama file asal tiap epoch, urutan identik dengan load_epochs (untuk analisis kebocoran & EDA).
    Hanya membaca panjang file; jitter direproduksi dengan seed yang sama."""
    out = []
    for p in sorted(os.listdir(data_dir), key=numerical_sort):
        if not p.endswith('.wav') or p.split('_')[0] == 'An.Minimus':
            continue
        n = sf.info(os.path.join(data_dir, p)).frames
        for i in range(0, n, TARGET // 2):
            np.random.seed(i)
            if i + np.random.randint(120) + TARGET <= n:
                out.append(p)
    return out

# [lib] split-fold
# Kuota test per kelompok (hanya mikrofon low-cost), disalin apa adanya dari notebook rilis.
COUNT_TEST_LOW = {
    'Cx.Quin_1F_14D_LowMic': 97, 'Ae.Aegypti_1F_21D_LowMic': 19,
    'An.Dirus_1M_14D_LowMic': 30, 'An.Dirus_1F_14D_LowMic': 34,
    'Ae.Aegypti_1M_21D_LowMic': 39, 'Ae.Albopictus_1F_7D_LowMic': 51,
    'Ae.Albopictus_1M_7D_LowMic': 46, 'Cx.Quin_1M_14D_LowMic': 72,
    'Ae.Aegypti_1F_5D_LowMic': 64, 'Ae.Albopictus_1F_14D_LowMic': 43,
    'Ae.Albopictus_1M_14D_LowMic': 57, 'Ae.Aegypti_1M_5D_LowMic': 48,
}


def split_fold(epochs, totals, fold):
    """Port 1:1 split_data(): epoch tiap kelompok dibagi kronologis menjadi 10 potongan.
    Potongan k -> validation, potongan k+1 -> test (LowMic saja, dibatasi kuota), sisanya -> train."""
    test_index = -1 if fold == 9 else fold
    seen = {k: 0 for k in totals}
    low_count = {k: 0 for k in COUNT_TEST_LOW}
    sets = {s: ([], []) for s in ('train', 'val', 'test')}
    for key, label, low, y in epochs:
        cur = seen[key]
        seen[key] += 1
        chunk = math.floor(totals[key] * 0.1)
        if fold * chunk <= cur < (fold + 1) * chunk:
            dst = 'val'
        elif (test_index + 1) * chunk <= cur < (test_index + 2) * chunk:
            if low and low_count[key] < COUNT_TEST_LOW[key]:
                low_count[key] += 1
                dst = 'test'
            else:
                continue                            # sisa potongan test tidak dipakai
        else:
            dst = 'train'
        sets[dst][0].append(y)
        sets[dst][1].append(label)
    return {s: (np.stack(X).astype(np.float32), np.array(Y)) for s, (X, Y) in sets.items()}


def encode(labels):
    idx = {c: i for i, c in enumerate(CLASSES)}
    return np.array([idx[l] for l in labels], dtype=np.int64)

# [lib] simulasi-noise
C_SOUND = 343.0
_LEN = 0.3                                            # jarak M1-M2 (m)
_M1, _M2, _SRC = (-_LEN / 2, 0.0), (_LEN / 2, 0.0), (_LEN * 0.5, 0.05)
D1 = math.hypot(_M1[0] - _SRC[0], _M1[1] - _SRC[1])  # ~0.304 m
D2 = math.hypot(_M2[0] - _SRC[0], _M2[1] - _SRC[1])  # 0.05 m
DELAY1, DELAY2 = int(D1 / C_SOUND * SR), int(D2 / C_SOUND * SR)   # 7 dan 1 sampel


def load_noise(noise_dir):
    """List (kanal_M1, kanal_M2); urutan file = sorted(os.listdir) seperti notebook rilis."""
    out, names = [], sorted(os.listdir(noise_dir))
    for fn in names:
        x, sr = sf.read(os.path.join(noise_dir, fn), dtype='float32', always_2d=True)
        assert sr == SR and x.shape[1] == 2
        out.append((x[:, 1].copy(), x[:, 0].copy()))   # separate_noise_channels(): (ch1, ch0)
    return out, names


def _delay(x, d):
    """Geser d sampel ke kanan (zero padding) dengan panjang tetap."""
    if d == 0:
        return x
    y = np.zeros_like(x)
    y[..., d:] = x[..., :-d]
    return y


def mosquito_cancelled(x, g):
    """Sinyal nyamuk setelah noise cancellation: (G/d2)·x(t-1) - (G/d1)·x(t-7)."""
    return g / D2 * _delay(x, DELAY2) - g / D1 * _delay(x, DELAY1)


def overlay_one(x, f, seed, noise):
    """Port OverlayAndDelayPerBatchTestSet(is_noise_cancelled=True), termasuk seeding `random`."""
    random.seed(seed)
    n1, n2 = noise[random.randint(0, len(noise) - 1)]
    clean = mosquito_cancelled(x[None], f)[0]
    random.seed(seed)
    start = int(round(random.uniform(0, len(n1) - TARGET), 0))
    noisy = clean + (n2[start:start + TARGET] - n1[start:start + TARGET])
    return random.choice([clean, noisy])               # catatan: 50% set "noisy" tetap bersih


def make_noisy_set(X, seed_offset, noise, gains=GAINS):
    """create_validation_or_test_set(): dict G -> array (N, 2400)."""
    return {g: np.stack([overlay_one(x, g, seed_offset + i, noise) for i, x in enumerate(X)]).astype(np.float32)
            for g in gains}


def make_clean_set(X, gains=GAINS):
    """create_test_set_without_noise(): DelayTestSet dengan noise cancellation."""
    return {g: mosquito_cancelled(X, g).astype(np.float32) for g in gains}

# [lib] model-mosquitosong
def mosquitosong(n_classes=8, input_len=2400):
    """MosquitoSong (Fig. 2 kiri): down-sampling dengan max-pool."""
    inp = L.Input((input_len, 1))
    x = L.Conv1D(32, 350, strides=4, activation='relu')(inp)
    x = L.MaxPooling1D(4, strides=4)(x)
    x = L.Conv1D(32, 100, strides=1, activation='relu')(x)
    x = L.MaxPooling1D(3, strides=3)(x)
    x = L.Dropout(0.5)(x)
    x = L.Flatten()(x)
    x = L.Dense(256, activation='relu')(x)
    x = L.Dense(128, activation='relu')(x)
    out = L.Dense(n_classes, activation='softmax', dtype='float32')(x)
    return keras.Model(inp, out, name='MosquitoSong')


def mosquitosong_plus(n_classes=8, input_len=2400):
    """MosquitoSong+ (Fig. 2 kanan): tiga conv bertingkat sebagai down-sampling yang dipelajari."""
    inp = L.Input((input_len, 1))
    x = L.Conv1D(32, 100, strides=4, activation='relu')(inp)
    x = L.Conv1D(32, 64, strides=4, activation='relu')(x)
    x = L.Conv1D(64, 64, strides=3, activation='relu')(x)
    x = L.MaxPooling1D(3, strides=3)(x)
    x = L.Dropout(0.5)(x)
    x = L.Flatten()(x)
    x = L.Dense(256, activation='relu')(x)
    x = L.Dense(128, activation='relu')(x)
    out = L.Dense(n_classes, activation='softmax', dtype='float32')(x)
    return keras.Model(inp, out, name='MosquitoSongPlus')


MODELS = {'mosquitosong': mosquitosong, 'mosquitosong_plus': mosquitosong_plus}

# [lib] augmentasi
def augment_batch(X, rng, noise=None, noise_aug=True, p_noise=0.9, gain_range=(1.0, 2.0)):
    """Augmentasi baru untuk setiap contoh di setiap epoch (vektorisasi numpy):
    1) time shift sirkular ±10 %; 2) volume G~U(1,2) [noise_aug]; 3) simulasi 2-mic + cancellation;
    4) noise N-OUTDOOR p=0.9 [noise_aug]; 5) amplitudo -12..12 dB; 6) Gaussian U(0.001,0.01)."""
    n = len(X)
    shifts = rng.integers(-TARGET // 10, TARGET // 10 + 1, size=n)
    idx = (np.arange(TARGET)[None, :] - shifts[:, None]) % TARGET
    x = np.take_along_axis(X, idx, axis=1)

    g = rng.uniform(*gain_range, size=(n, 1)).astype(np.float32) if noise_aug else np.ones((n, 1), np.float32)
    x = mosquito_cancelled(x, g)

    if noise_aug and noise:
        add = rng.random(n) < p_noise
        files = rng.integers(0, len(noise), size=n)
        for i in np.where(add)[0]:
            n1, n2 = noise[files[i]]
            s = rng.integers(0, len(n1) - TARGET + 1)
            x[i] += n2[s:s + TARGET] - n1[s:s + TARGET]

    x *= (10 ** (rng.uniform(-12, 12, size=(n, 1)) / 20)).astype(np.float32)
    x += rng.uniform(0.001, 0.01, size=(n, 1)).astype(np.float32) * rng.standard_normal(x.shape).astype(np.float32)
    return x.astype(np.float32)

# [lib] training-per-fold
DEFAULT_CFG = dict(model='mosquitosong_plus', fold=0, epochs=1000, batch=64, lr=1e-4, seed=42,
                   recipe='paper', patience=0, tag='', precision='fp32', no_noise_aug=False, pretrained=True,
                   data='data', out='runs', log_every=10, ckpt_every=10, deadline=None, pause_at=0, max_train=0)
PRED_BATCH = 256          # batch inferensi (cukup kecil untuk DenseNet121 128x128 di T4)
PRETRAINED_MODELS = set() # model yang menerima argumen pretrained=... (diisi sel DenseNet121)


def oversample(X, y, random_state=42):
    """RandomOverSampler(sampling_strategy='all', random_state=42) seperti notebook rilis;
    fallback numpy setara bila imbalanced-learn tidak terpasang."""
    try:
        from imblearn.over_sampling import RandomOverSampler
        return RandomOverSampler(sampling_strategy='all', random_state=random_state).fit_resample(X, y)
    except ImportError:
        rng = np.random.default_rng(random_state)
        n_max = np.bincount(y).max()
        idx = np.concatenate([np.concatenate([np.where(y == c)[0], rng.choice(np.where(y == c)[0], n_max - (y == c).sum())])
                              for c in np.unique(y)])
        return X[idx], y[idx]


def setup_device(precision):
    for gpu in tf.config.list_physical_devices('GPU'):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass                                    # GPU sudah diinisialisasi (fold kedua in-process)
    keras.mixed_precision.set_global_policy({'bf16': 'mixed_bfloat16', 'fp16': 'mixed_float16'}.get(precision, 'float32'))
    gpus = tf.config.list_physical_devices('GPU')
    return 'GPU:' + ','.join(tf.config.experimental.get_device_details(g).get('device_name', '?') for g in gpus) \
        if gpus else 'CPU:' + (platform.processor() or platform.machine())


def build_model(name, pretrained=True):
    fn = MODELS[name]
    return fn(len(CLASSES), pretrained=pretrained) if name in PRETRAINED_MODELS else fn(len(CLASSES))


def save_checkpoint(ckpt_dir, model, rng, state):
    """Checkpoint tengah-fold: bobot terakhir + state optimizer + state RNG augmentasi + histori.
    Semua file ditulis lewat nama sementara lalu diganti; state.json ditulis terakhir (penanda checkpoint valid)."""
    os.makedirs(ckpt_dir, exist_ok=True)
    model.save_weights(os.path.join(ckpt_dir, 'last.tmp.weights.h5'))
    os.replace(os.path.join(ckpt_dir, 'last.tmp.weights.h5'), os.path.join(ckpt_dir, 'last.weights.h5'))
    opt = {f'v{i}': keras.ops.convert_to_numpy(v) for i, v in enumerate(model.optimizer.variables)}
    np.savez(os.path.join(ckpt_dir, 'optimizer.tmp.npz'), **opt)
    os.replace(os.path.join(ckpt_dir, 'optimizer.tmp.npz'), os.path.join(ckpt_dir, 'optimizer.npz'))
    with open(os.path.join(ckpt_dir, 'state.tmp.json'), 'w') as f:
        json.dump({**state, 'rng': rng.bit_generator.state}, f, default=str)
    os.replace(os.path.join(ckpt_dir, 'state.tmp.json'), os.path.join(ckpt_dir, 'state.json'))


def load_checkpoint(ckpt_dir, model, rng):
    with open(os.path.join(ckpt_dir, 'state.json')) as f:
        state = json.load(f)
    model.optimizer.build(model.trainable_variables)   # bangun optimizer dulu agar bobotnya bisa diisi
    model.load_weights(os.path.join(ckpt_dir, 'last.weights.h5'))
    saved = np.load(os.path.join(ckpt_dir, 'optimizer.npz'))
    if len(saved.files) == len(model.optimizer.variables):
        for i, v in enumerate(model.optimizer.variables):
            v.assign(saved[f'v{i}'])
    elif state['epoch'] > 0:                           # jeda di epoch 0: belum ada state optimizer
        print('Peringatan: state optimizer tidak cocok, optimizer dimulai ulang', flush=True)
    rng.bit_generator.state = state.pop('rng')
    return state


def run_fold(cfg):
    """Latih + evaluasi satu (model, fold). Artefak: runs/<nama>/fold<k>/
    best.weights.h5, history.csv, preds.npz (probabilitas test + validation), config.json (penanda fold lengkap).
    Selama training, ckpt/ diperbarui tiap `ckpt_every` epoch. Fold dijeda (checkpoint disimpan) bila
    `deadline` (waktu absolut) terlewati atau epoch mencapai `pause_at` (pilot); sesi berikutnya melanjutkan."""
    a = SimpleNamespace(**{**DEFAULT_CFG, **cfg})
    run_name = a.model + (f'_{a.tag}' if a.tag else '')
    out_dir = os.path.join(a.out, run_name, f'fold{a.fold}')
    ckpt_dir = os.path.join(out_dir, 'ckpt')
    os.makedirs(out_dir, exist_ok=True)
    seed = a.seed + a.fold
    keras.utils.set_random_seed(seed)
    device = setup_device(a.precision)
    rng = np.random.default_rng(seed)

    epochs_ = load_epochs(os.path.join(a.data, 'mosquito_recordings'))
    noise, noise_names = load_noise(os.path.join(a.data, 'environmental_noise_recordings'))
    s = split_fold(epochs_, count_per_group(epochs_), a.fold)
    Xtr, ytr = oversample(s['train'][0], encode(s['train'][1]))
    Xva, yva = s['val'][0], encode(s['val'][1])
    Xte, yte = s['test'][0], encode(s['test'][1])
    if a.max_train:                                     # hanya untuk uji cepat lokal
        sub = np.random.default_rng(0).choice(len(ytr), min(a.max_train, len(ytr)), replace=False)
        Xtr, ytr = Xtr[sub], ytr[sub]

    va = make_noisy_set(Xva, 2500, noise)               # validation: noise overlay G=1,1.5,2 (seed 2500)
    Xva_all, yva_all = np.concatenate([va[g] for g in GAINS])[..., None], np.tile(yva, len(GAINS))

    noise_aug = a.model != 'mosquitosong' and not a.no_noise_aug   # MosquitoSong lama: tanpa noise aug
    model = build_model(a.model, a.pretrained)
    steps = int(np.ceil(len(ytr) / a.batch)) * a.epochs
    lr = keras.optimizers.schedules.CosineDecay(a.lr, steps, alpha=0.01) if a.recipe in ('fast', 'finetune') else a.lr
    model.compile(optimizer=keras.optimizers.Adam(lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    hist, best, best_ep, start_ep, elapsed0 = [], np.inf, -1, 0, 0.0
    weights_path = os.path.join(out_dir, 'best.weights.h5')
    if os.path.exists(os.path.join(ckpt_dir, 'state.json')):          # lanjutkan fold yang terjeda
        st = load_checkpoint(ckpt_dir, model, rng)
        hist, best, best_ep, start_ep, elapsed0 = st['hist'], st['best'], st['best_ep'], st['epoch'], st['elapsed']
        print(f'[{run_name} f{a.fold}] resume dari checkpoint: epoch {start_ep}, best@{best_ep}', flush=True)
    t0 = time.time() - elapsed0

    def state(epoch_done):
        return dict(epoch=epoch_done, hist=hist, best=best, best_ep=best_ep, elapsed=time.time() - t0)

    def pause(ep, why):
        save_checkpoint(ckpt_dir, model, rng, state(ep))
        print(f'[{run_name} f{a.fold}] dijeda di epoch {ep} ({why}); dilanjutkan dari checkpoint', flush=True)
        return {'status': 'paused', 'epoch': ep}

    print(f'[{run_name} f{a.fold}] device={device} lr={a.lr:g} train={len(ytr)} val={len(yva_all)} test={len(yte)} '
          f'params={model.count_params():,}', flush=True)
    for ep in range(start_ep, a.epochs):
        if a.pause_at and ep >= a.pause_at:
            return pause(ep, f'pilot {a.pause_at} epoch')
        if a.deadline and time.time() > a.deadline:
            return pause(ep, 'batas waktu sesi')
        Xa = augment_batch(Xtr, rng, noise, noise_aug=noise_aug)[..., None]
        h = model.fit(Xa, ytr, batch_size=a.batch, epochs=1, shuffle=True, verbose=0)
        vl, vacc = model.evaluate(Xva_all, yva_all, batch_size=PRED_BATCH, verbose=0)
        hist.append(dict(epoch=ep + 1, loss=h.history['loss'][0], acc=h.history['accuracy'][0],
                         val_loss=vl, val_acc=vacc, t=time.time() - t0))
        if vl < best:                                   # pemilihan model: val loss minimum
            best, best_ep = vl, ep + 1
            model.save_weights(weights_path)
        if (ep + 1) % a.log_every == 0 or ep == start_ep:
            print(f'[{run_name} f{a.fold}] ep {ep+1}/{a.epochs} loss {h.history["loss"][0]:.4f} '
                  f'val_loss {vl:.4f} val_acc {vacc:.4f} best@{best_ep} ({time.time()-t0:.0f}s)', flush=True)
        if a.patience and ep + 1 - best_ep >= a.patience:
            print(f'[{run_name} f{a.fold}] early stop at epoch {ep+1}', flush=True)
            break
        if a.ckpt_every and (ep + 1) % a.ckpt_every == 0:
            save_checkpoint(ckpt_dir, model, rng, state(ep + 1))
    pd.DataFrame(hist).to_csv(os.path.join(out_dir, 'history.csv'), index=False)

    model.load_weights(weights_path)                    # test: protokol rilis (noisy + clean, tiap G)
    te_noisy, te_clean = make_noisy_set(Xte, 0, noise), make_clean_set(Xte)
    preds = {'y': yte, 'y_val': yva_all, 'val': model.predict(Xva_all, batch_size=PRED_BATCH, verbose=0)}
    for g in GAINS:
        preds[f'noisy_G{g}'] = model.predict(te_noisy[g][..., None], batch_size=PRED_BATCH, verbose=0)
        preds[f'clean_G{g}'] = model.predict(te_clean[g][..., None], batch_size=PRED_BATCH, verbose=0)
    np.savez_compressed(os.path.join(out_dir, 'preds.npz'), **preds)
    acc = float(np.mean([(preds[f'{c}_G{g}'].argmax(1) == yte).mean() for c in ('noisy', 'clean') for g in GAINS]))

    meta = dict(vars(a), run=run_name, seed_effective=seed, best_epoch=best_ep, best_val_loss=best,
                epochs_run=len(hist), resumed_from_epoch=start_ep, n_train_balanced=len(ytr), n_val=len(yva),
                n_test=len(yte), noise_aug=noise_aug, noise_files=noise_names, classes=CLASSES, device=device,
                params=model.count_params(), tf=tf.__version__, keras=keras.__version__, numpy=np.__version__,
                python=platform.python_version(), train_time_s=time.time() - t0, test_acc_all=acc)
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:      # ditulis terakhir = penanda fold lengkap
        json.dump(meta, f, indent=2, default=str)
    shutil.rmtree(ckpt_dir, ignore_errors=True)        # checkpoint tidak diperlukan lagi
    print(f'[{run_name} f{a.fold}] done. best epoch {best_ep}, test ACC {acc:.3f}, '
          f'{(time.time()-t0)/60:.1f} min', flush=True)
    return {'status': 'done', **meta}

# [lib] pembanding-blok-umum
def conv_bn(x, ch, k, s=1, d=1, act=True):
    x = L.Conv1D(ch, k, strides=s, dilation_rate=d, padding='same', use_bias=False)(x)
    x = L.BatchNormalization()(x)
    return L.ReLU()(x) if act else x


def head(x, n_classes):
    """Kepala klasifikasi bersama: dropout 0,5 -> FC 128 -> softmax (fp32 walau mixed precision)."""
    x = L.Dropout(0.5)(x)
    x = L.Dense(128, activation='relu')(x)
    return L.Dense(n_classes, activation='softmax', dtype='float32')(x)

# [lib] fitur-spektral
class MFCC(L.Layer):
    """MFCC dihitung di dalam model (tf.signal): selalu dari waveform yang sudah diaugmentasi, ikut berjalan di GPU.
    Frame 512 (64 ms, 15,6 Hz/bin), hop 64 (8 ms) -> 30 frame; 64 pita mel 100-4000 Hz; 40 koefisien c1..c40.
    c0 (energi) dibuang sehingga fitur tidak peka augmentasi amplitudo -12..12 dB."""

    def __init__(self, n_mfcc=40, frame=512, hop=64, n_mels=64, fmin=100.0, fmax=4000.0, **kw):
        kw.setdefault('dtype', 'float32')           # STFT selalu fp32 walau mixed precision
        super().__init__(**kw)
        self.n_mfcc, self.frame, self.hop, self.n_mels, self.fmin, self.fmax = n_mfcc, frame, hop, n_mels, fmin, fmax

    def call(self, x):
        x = tf.cast(tf.squeeze(x, -1), tf.float32)
        spec = tf.abs(tf.signal.stft(x, self.frame, self.hop, self.frame))                 # (B, 30, 257)
        mel = tf.signal.linear_to_mel_weight_matrix(self.n_mels, self.frame // 2 + 1, SR, self.fmin, self.fmax)
        logmel = tf.math.log(tf.tensordot(spec, mel, 1) + 1e-6)                            # (B, 30, 64)
        return tf.signal.mfccs_from_log_mel_spectrograms(logmel)[..., 1:self.n_mfcc + 1]  # (B, 30, 40)

    def compute_output_shape(self, shape):
        return (shape[0], (shape[1] - self.frame) // self.hop + 1, self.n_mfcc)


class LogSpectrogram(L.Layer):
    """Log-spektrogram sebagai 'gambar' untuk CNN ImageNet: STFT 512 / hop 32 -> 60 frame x 251 bin (±100-4000 Hz;
    6 bin terbawah dibuang karena berisi rumble < 100 Hz yang mendominasi energi, lihat EDA 5.6), log-magnitude,
    standardisasi per sampel, frekuensi rendah di bawah, resize 128 x 128, diulang ke 3 kanal."""

    def __init__(self, frame=512, hop=32, size=128, min_bin=6, **kw):
        kw.setdefault('dtype', 'float32')
        super().__init__(**kw)
        self.frame, self.hop, self.size, self.min_bin = frame, hop, size, min_bin

    def call(self, x):
        x = tf.cast(tf.squeeze(x, -1), tf.float32)
        s = tf.abs(tf.signal.stft(x, self.frame, self.hop, self.frame))[..., self.min_bin:]   # (B, 60, 251)
        s = tf.math.log(s + 1e-6)
        mu = tf.reduce_mean(s, axis=[1, 2], keepdims=True)
        sd = tf.math.reduce_std(s, axis=[1, 2], keepdims=True) + 1e-6
        s = tf.reverse(tf.transpose((s - mu) / sd, [0, 2, 1]), axis=[1])[..., None]          # (B, 251, 60, 1)
        s = tf.image.resize(s, (self.size, self.size))
        return tf.repeat(s, 3, axis=-1)                                                       # (B, 128, 128, 3)

    def compute_output_shape(self, shape):
        return (shape[0], self.size, self.size, 3)

# [lib] pembanding-ecapa-mfcc
def se_res2_block(x, ch, k, d, scale=4, se_bottleneck=64):
    """SE-Res2Net: channel dibagi `scale` grup, konvolusi hierarkis (multi-skala), SE, residual."""
    res = x
    splits = ops.split(conv_bn(x, ch, 1), scale, axis=-1)
    outs, prev = [splits[0]], None
    for i in range(1, scale):
        s = splits[i] if prev is None else L.Add()([splits[i], prev])
        prev = conv_bn(s, ch // scale, k, d=d)
        outs.append(prev)
    h = conv_bn(L.Concatenate()(outs), ch, 1)
    w = L.GlobalAveragePooling1D(keepdims=True)(h)
    w = L.Conv1D(ch, 1, activation='sigmoid')(L.Conv1D(se_bottleneck, 1, activation='relu')(w))
    return L.Add()([L.Multiply()([h, w]), res])


class AttentiveStatsPool(L.Layer):
    """Attentive Statistics Pooling: bobot per posisi & per channel -> [mean berbobot ‖ std berbobot]."""

    def __init__(self, bottleneck=64, **kw):
        super().__init__(**kw)
        self.bottleneck = bottleneck

    def build(self, shape):
        ch = shape[-1]
        self.att1 = L.Conv1D(self.bottleneck, 1, activation='tanh')
        self.att2 = L.Conv1D(ch, 1)
        self.att1.build((None, None, ch * 3))
        self.att2.build((None, None, self.bottleneck))

    def call(self, x):
        x = ops.cast(x, 'float32')
        mu = ops.mean(x, axis=1, keepdims=True)
        sd = ops.sqrt(ops.clip(ops.var(x, axis=1, keepdims=True), 1e-5, 1e5))
        t = ops.shape(x)[1]
        ctx = ops.concatenate([x, ops.repeat(mu, t, axis=1), ops.repeat(sd, t, axis=1)], -1)
        a = ops.softmax(ops.cast(self.att2(self.att1(ops.cast(ctx, self.compute_dtype))), 'float32'), axis=1)
        m = ops.sum(a * x, axis=1)
        s = ops.sqrt(ops.clip(ops.sum(a * x * x, axis=1) - m * m, 1e-5, 1e5))
        return ops.concatenate([m, s], -1)

    def compute_output_shape(self, shape):
        return (shape[0], 2 * shape[-1])


def ecapa_mfcc(n_classes=8, input_len=2400, ch=96):
    """ECAPA-TDNN dengan input MFCC seperti paper aslinya (Desplanques et al., 2020), versi kecil."""
    inp = L.Input((input_len, 1))
    x = L.BatchNormalization()(MFCC()(inp))         # 30 frame x 40 koefisien, dinormalisasi per koefisien
    x = conv_bn(x, ch, 5)                           # 30 x 96
    b1 = se_res2_block(x, ch, 3, 2)
    b2 = se_res2_block(b1, ch, 3, 3)
    b3 = se_res2_block(b2, ch, 3, 4)
    x = conv_bn(L.Concatenate()([b1, b2, b3]), 3 * ch, 1)   # Multi-layer Feature Aggregation
    x = L.BatchNormalization()(AttentiveStatsPool()(x))     # 576
    return keras.Model(inp, head(x, n_classes), name='ECAPA_MFCC')

# [lib] pembanding-mfcc
def mfcc_cnn(n_classes=8, input_len=2400):
    """Pembanding 'fitur beda': hanya MFCC (30 frame x 40 koefisien) -> CNN 1D kecil."""
    inp = L.Input((input_len, 1))
    x = L.BatchNormalization()(MFCC()(inp))         # normalisasi tiap koefisien
    x = conv_bn(conv_bn(x, 64, 3), 64, 3)
    x = L.MaxPooling1D(2)(x)                        # 15 frame
    x = conv_bn(conv_bn(x, 128, 3), 128, 3)
    x = L.GlobalAveragePooling1D()(x)
    return keras.Model(inp, head(x, n_classes), name='MFCC_CNN')


def mosquitosong_plus_mfcc(n_classes=8, input_len=2400):
    """Fusion: cabang waveform MosquitoSong+ (utuh sampai Flatten) + cabang statistik MFCC (mean & std tiap koefisien)."""
    inp = L.Input((input_len, 1))
    x = L.Conv1D(32, 100, strides=4, activation='relu')(inp)
    x = L.Conv1D(32, 64, strides=4, activation='relu')(x)
    x = L.Conv1D(64, 64, strides=3, activation='relu')(x)
    x = L.Flatten()(L.Dropout(0.5)(L.MaxPooling1D(3, strides=3)(x)))            # 448
    m = MFCC()(inp)
    m = L.Concatenate()([ops.mean(m, axis=1), ops.std(m, axis=1)])              # 80
    m = L.Dense(64, activation='relu')(L.BatchNormalization()(m))
    h = L.Concatenate()([x, m])
    h = L.Dense(128, activation='relu')(L.Dense(256, activation='relu')(h))
    out = L.Dense(n_classes, activation='softmax', dtype='float32')(h)
    return keras.Model(inp, out, name='MosquitoSongPlus_MFCC')


MODELS.update({'mfcc_cnn': mfcc_cnn, 'ecapa_mfcc': ecapa_mfcc, 'mosquitosong_plus_mfcc': mosquitosong_plus_mfcc})

# [lib] pembanding-densenet121
def densenet121_spec(n_classes=8, input_len=2400, pretrained=True, asp=False):
    """DenseNet121 (bobot ImageNet) pada log-spektrogram 128 x 128 x 3.
    asp=False: global average pooling (standar). asp=True: ablation attention, peta fitur terakhir 4 x 4 x 1024
    diperlakukan sebagai 16 posisi waktu-frekuensi lalu diringkas dengan Attentive Statistics Pooling."""
    inp = L.Input((input_len, 1))
    x = LogSpectrogram()(inp)
    base = keras.applications.DenseNet121(include_top=False, weights='imagenet' if pretrained else None,
                                          input_shape=(128, 128, 3))
    f = base(x)                                     # (4, 4, 1024)
    if asp:
        f = L.BatchNormalization()(AttentiveStatsPool()(L.Reshape((16, 1024))(f)))   # 2048
    else:
        f = L.GlobalAveragePooling2D()(f)           # 1024
    return keras.Model(inp, head(f, n_classes), name='DenseNet121_ASP' if asp else 'DenseNet121')


def densenet121(n_classes=8, input_len=2400, pretrained=True):
    return densenet121_spec(n_classes, input_len, pretrained, asp=False)


def densenet121_asp(n_classes=8, input_len=2400, pretrained=True):
    return densenet121_spec(n_classes, input_len, pretrained, asp=True)


MODELS.update({'densenet121': densenet121, 'densenet121_asp': densenet121_asp})
PRETRAINED_MODELS.update({'densenet121', 'densenet121_asp'})


if __name__ == '__main__':
    run_fold(json.loads(sys.argv[1]))
