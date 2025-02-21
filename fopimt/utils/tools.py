import io
import os
import zipfile
import logging


def get_zip_buffer(directory_path: str) -> io.BytesIO | None:
    # Create an in-memory bytes buffer
    zip_buffer = io.BytesIO()

    # Create a ZIP file in memory
    try:
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for root, _, files in os.walk(directory_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, start=directory_path)
                    zip_file.write(file_path, arcname)
    except Exception as e:
        logging.error(f"Cannot create zip buffer: {e}.")
        return None

    # Seek to the beginning of the BytesIO buffer
    zip_buffer.seek(0)

    return zip_buffer
