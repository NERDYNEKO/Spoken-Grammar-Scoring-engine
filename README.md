# Spoken Grammar Scoring Engine

A machine learning pipeline that takes a 45-60 second speech recording and predicts a continuous **grammar score from 0 to 5** (MOS Likert rubric).

## Results

| Metric | Value |
|---|---|
| Training RMSE | 0.2175 |
| Training Pearson | 0.9846 |
| **5-fold CV RMSE** (unseen-data estimate) | **0.5311** |
| **5-fold CV Pearson** | **0.9035** |
| Baseline RMSE (predict the mean) | 1.2382 |

The stacked model beats every single model. Training RMSE is lower than CV RMSE because the final models have seen the training files, so the CV numbers are the honest estimate.

## Approach

1. **Preprocessing:** audio loaded as mono, resampled to 16 kHz, trimmed to 60 s, split into 30 s chunks for Whisper.
2. **Speech-to-text:** Whisper-small generates an English transcript.
3. **Features (3 groups):**
   - **Audio embedding:** Whisper encoder output (middle + last layer), averaged over time.
   - **Text embedding:** RoBERTa-base embedding of the transcript.
   - **Grammar and fluency statistics:** GPT-2 loss (higher means less grammatical), words per minute, sentence lengths, filler words, repeated words.
4. **Models:** Ridge and SVR on each feature group, plus Gradient Boosting on the statistics.
5. **Stacking:** a non-negative linear blend of 5-fold out-of-fold predictions, clipped to [0, 5].
