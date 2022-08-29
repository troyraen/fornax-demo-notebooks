# This script syncs Jupyter notebooks with representations in other
# file formats (such as .md) using the tool jupytext.

notebooks=(
    forced_photometry/multiband_photometry.ipynb
)

install () {
    reqs=(
        jupytext
        black==22.6.0
        mdformat-myst
        mdformat-config
        mdformat-frontmatter
        mdformat-web
        # pre-commit
    )
    pip install "${reqs[@]}"
}

setupsync () {
    formats="${1:-ipynb,myst}"
    for nb in "${notebooks[@]}"; do
        jupytext --set-formats "${formats}" "${nb}"
    done
}

run () {
    # sync (jupytext) all the notebooks, reformat markdowns, sync again
    _syncall
    _mdformatall
    _syncall
}

_mdformatall() {
    for nb in "${notebooks[@]}"; do
        # equivalent calls. not sure if we should use pre-commit or just mdformat directly
        # mdformat "${nb%.ipynb}.md"
        pre-commit run --files "${nb%.ipynb}.md"  # pre-commit gives a nice info message
    done

}

_syncall () {
    for nb in "${notebooks[@]}"; do
        jupytext "${nb}" --sync
        # jupytext "${nb}" --sync --execute
        # jupytext "${nb}" --sync --pipe black --pipe mdformat --pipe-fmt md
    done
}

help () {
    echo "This script syncs Jupyter notebooks with representations in other"
    echo "file formats (such as .md) using the tool jupytext."
    echo
    echo "Syntax: bash precommit.sh [--install|setupsync|help]"
    echo
    echo "Sync the registered notebooks with their paired representations"
    echo "by calling the script with no arguments."
    echo
    echo "Options:"
    echo
    echo "-h, --help    Print this message and exit."
    echo
    echo "--install     Install the required dependencies (only needs to be done once)."
    echo
    echo "--setupsync   Pair the registered notebooks with another file format(s)."
    echo "              The default pairing is ipynb,myst."
    echo "              Optionally, pass the desired formats (comma separated) as an"
    echo "              additional argument, for example:"
    echo "              $ bash precommit.sh --setupsync ipynb,myst"
    echo "              To register a new notebook, add the notebook's relative path to"
    echo "              the `notebooks` array at the top of this script (precommit.sh)."
    echo
}

if [ "${1}" = "-h"  -o "$1" = "--help" ]; then
    help
elif [ "${1}" = "--install" ]; then
    install
elif [ "${1}" = "--setupsync" ]; then
    setupsync "${2}"  # 2=sync formats (comma separated)
elif [ "${1}" = "-s" ]; then
    _syncall
elif [ "${1}" = "-m" ]; then
    _mdformatall
else
    run
fi

