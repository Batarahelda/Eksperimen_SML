from PIL import Image
import shutil
import os


def clean_dataset(input_dir, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    for cls in os.listdir(input_dir):

        source_class = os.path.join(input_dir, cls)
        target_class = os.path.join(output_dir, cls)

        os.makedirs(target_class, exist_ok=True)

        for file in os.listdir(source_class):

            try:

                source_file = os.path.join(
                    source_class,
                    file
                )

                target_file = os.path.join(
                    target_class,
                    file
                )

                img = Image.open(source_file)

                img.verify()

                shutil.copy(
                    source_file,
                    target_file
                )

            except Exception as e:

                print(
                    f"File rusak: {file}"
                )


if __name__ == "__main__":

    INPUT_DIR = "./plantvillage_raw"
    OUTPUT_DIR = "./plantvillage_preprocessing"

    if not os.path.exists(INPUT_DIR):

        print(
            f"Dataset tidak ditemukan: {INPUT_DIR}"
        )

        print(
            "Preprocessing dilewati."
        )

        exit(0)

    clean_dataset(
        INPUT_DIR,
        OUTPUT_DIR
    )