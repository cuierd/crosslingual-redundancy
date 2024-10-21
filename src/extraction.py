from src.data.components.feature_extractors import ProsodyFeatureExtractor


def extract():
    for category in ['train-clean-100','test-clean','dev-clean']:

        lang = 'zh'
        LAB_ROOT = f"/home/user/ding/Projects/Prosody/languages/{lang}/aligned/{category}"
        WAV_ROOT = f"/home/user/ding/Projects/Prosody/languages/{lang}/wav_files/{category}"
        DATA_CACHE = f"/home/user/ding/Projects/Prosody/languages/{lang}/cache"

        extractor = ProsodyFeatureExtractor(
        lab_root=LAB_ROOT,
        wav_root=WAV_ROOT,
        phoneme_lab_root=LAB_ROOT,
        data_cache=DATA_CACHE,
        language = lang,
        extract_f0=True,
        f0_stress_localizer = None,
        celex_path="",
        f0_n_coeffs=4,
        f0_mode='dct',
        )

    print('Feature extraction complete')
    return extractor


def main():
    extractor = extract()
    return extractor


if __name__ == "__main__":
    main()
