# TabWak repo: known gaps and fixes

## 1. Missing `data/Info/*.json` (blocks `process_dataset.py`)

The TabWak repo ships **no** dataset config templates, but
`process_dataset.py` reads `data/Info/<name>.json` and everything
downstream (`main.py`, `tabsyn/latent_utils.get_input_generate`) needs the
`data/<name>/info.json` that it writes. The configs live in the upstream
TabSyn repo (amazon-science/tabsyn, `data/Info/`); verbatim copies are in
`third_party/tabwak_data_Info/` here (paths already match TabWak's layout).

Fix, inside your TabWak checkout (raw data already downloaded):

    mkdir -p data/Info
    cp /path/to/tabular-pancakes/third_party/tabwak_data_Info/*.json data/Info/
    python process_dataset.py --dataname adult

## 2. Always pass `--dataname`

Bare `python process_dataset.py` iterates over the literal name 'all' and
crashes (`data/Info/all.json` does not exist). One dataset per call.

## 3. Order of operations (Adult)

    python download_dataset.py                       # raw zips (done already)
    python process_dataset.py --dataname adult       # needs fix 1
    python main.py --dataname adult --method vae    --mode train
    python main.py --dataname adult --method tabsyn --mode train

Checkpoints land in tabsyn/vae/ckpt/adult/{encoder,decoder,model}.pt and
tabsyn/ckpt/adult/model.pt -- exactly what pancakemark.hosts.TabSynHost
loads. Then run the inversion feasibility check from the TabSynHost docstring.

(Verified against TabWak@main with a synthetic mini-adult:
processing completes and writes info.json + npy/csv splits.)
