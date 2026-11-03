# LM_EncDec_Base_LuongLSTM_dot

Encoder-decoder translation model ported from legacy TF2 code C5 (dot) to PyTorch (cell=lstm, attention=dot).

## Status

- Verification level: **Benchmarked (LM-03, one run)**

## Changes from the legacy code

- PyTorch port; beam search reimplemented (TF Addons is no longer maintained)
- Vocabulary built from the training split only (legacy built it from all pairs)
- "what's" expanded to "what is" (legacy wrote "that is")

## LM-03 setting

English -> French, 5-11 words, 12,800 pairs (seed 1234), 80/20 split, embedding 128, hidden 1024, batch 256, 20 epochs, Adam 1e-3. Small data: compare structures, not absolute BLEU.

## Results

- Parameters: 25.1 M
- Test perplexity 49.6, next-word accuracy 0.509
- BLEU-4 greedy (whole test split): 14.5
- BLEU-4 beam width 4 on 500 pairs: 15.6 (greedy on the same pairs 14.2)
- Examples:
  - `this winter has been mild .` -> `cette histoire est ces derniers .` (reference `cet hiver a ete doux .`)
  - `they will be here at three .` -> `ils vont rester ici .` (reference `ils seront ici a trois heures .`)
  - `i once lived in rome .` -> `je vis dans une experience .` (reference `j'ai vecu un temps a rome .`)

## Run

```bash
pip install torch==2.5.1 einops numpy
# put fra-eng.zip from https://www.manythings.org/anki/ into data/
python run.py lm03 s2s_luong_lstm_dot
```
