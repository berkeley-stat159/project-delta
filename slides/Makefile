.PHONY: all clean

all: clean progress.pdf

clean:
	rm -f progress.pdf

progress.pdf: progress.md
	pandoc -t beamer -s progress.md -o progress.pdf
final.pdf: final.md
	pandoc -t beamer -s final.md -o final.pdf
