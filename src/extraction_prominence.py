from src.data.components.feature_extractors import ProsodyFeatureExtractor


def extract():
    for category in ['train-clean-100','test-clean','dev-clean']:

        lang = 'en'
        LAB_ROOT = f"/Users/cui/Documents/uzh/PhD/Projects/Prosody/crosslingual-redundancy/languages/{lang}/aligned/{category}"
        WAV_ROOT = f"/Users/cui/Documents/uzh/PhD/Projects/Prosody/crosslingual-redundancy/languages/{lang}/wav_files/{category}"
        DATA_CACHE = f"/Users/cui/Documents/uzh/PhD/Projects/Prosody/crosslingual-redundancy/languages/{lang}/cache_prominence"

        # extractor = ProsodyFeatureExtractor(
        # lab_root=LAB_ROOT,
        # wav_root=WAV_ROOT,
        # phoneme_lab_root=LAB_ROOT,
        # data_cache=DATA_CACHE,
        # language = lang,
        # extract_f0=True,
        # f0_stress_localizer = None,
        # celex_path="",
        # f0_n_coeffs=4,
        # f0_mode='curve',#"dct" for paramarized
        # )

        extractor = ProsodyFeatureExtractor(
        lab_root=LAB_ROOT,
        wav_root=WAV_ROOT,
        phoneme_lab_root=LAB_ROOT,
        data_cache=DATA_CACHE,
        language = lang,
        extract_f0=False,
        extract_prominence=True,
        prominence_mode='mean'
        )

    print('Feature extraction complete.')
    return extractor


def main():
    extractor = extract()
    return extractor


if __name__ == "__main__":
    main()
