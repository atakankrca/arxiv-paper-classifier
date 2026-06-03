import fire

from arxiv_paper_classifier.commands import download, infer, preprocess, train


def main() -> None:
    fire.Fire(
        {
            "train": train,
            "infer": infer,
            "download": download,
            "preprocess": preprocess,
        }
    )


if __name__ == "__main__":
    main()
