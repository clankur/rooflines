.PHONY: dev build clean

dev:
	marimo edit rooflines_app.py

build:
	marimo export html-wasm rooflines_app.py -o docs --mode run --show-code --force
	cp accelerators.json docs/

clean:
	rm -rf docs/
