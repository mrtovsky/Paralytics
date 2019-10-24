import os
import warnings

from pathlib import Path

try:
    from pydrive.auth import GoogleAuth
    from pydrive.drive import GoogleDrive
except ImportError as err:
    _HAS_PYDRIVE = err
else:
    _HAS_PYDRIVE = True


__all__ = [
    "upload_to_google_drive"
]


def upload_to_google_drive(client_secrets_path, files_paths, folder_id="root",
                           credentials_path=None, save_credentials=False,
                           verbose=False):
    """Upload specified files to Google Drive.

    Support uploading files to Google Drive from Python after the project has
    been created in Google Developers Console. Guide explaining the
    configuration of the project is available on `medium.com
    <hhttps://medium.com/@annissouames99/how-to-upload-files-automatically-to-drive-with-python-ee19bb13dda/>`_.

    After successful configuration you should save your client ID and the
    secret key to the ``client_secrets.json`` file.

    Parameters
    ----------
    client_secrets_path : string or pathlib.Path
        Path to ``client_secrets.json`` file where client ID and the secret key
        for the project created via Google Developers Console are saved.

    files_paths : string or pathlib.Path or list of those objects
        Paths to files that will be uploaded to Google Drive.

    folder_id : string, optional (default="root")
        Receiving Google Drive folder id.

    credentials_path : string, optional (default=None)
        Path to the credentials json file where your Google credentials are or
        will be saved. Allows to avoid manual authentication every time files
        are uploaded which leads to full automation of this process.

    save_credentials : optional (default=False)
        Whether to save credentials to file specified in the
        ``credentials_path`` parameter.

    verbose : optional (default=False)
        Turn on/off function verbosity.

    """
    if isinstance(_HAS_PYDRIVE, ImportError):
        raise ImportError(
            "`upload_to_google_drive` requires extra requirements installed. "
            "Upgrade paralytics package with 'gdrive' extra specified or "
            "install the dependencies directly from the source."
        ).with_traceback(err.__traceback__)

    client_secrets_path = Path(client_secrets_path).resolve()

    if not client_secrets_path.is_file():
        raise FileNotFoundError(
            "Calling `upload` function requires `client_secrets.json` "
            "file existing.\nRead the docstring to make yourself familiar with "
            "acquiring this file."
        )

    GoogleAuth.DEFAULT_SETTINGS["client_config_file"] = str(client_secrets_path)

    gauth = GoogleAuth()

    if credentials_path is not None:
        credentials_path = Path(credentials_path).resolve()
        gauth.LoadCredentialsFile(str(credentials_path))
    elif save_credentials:
        save_credentials = False
        warnings.warn(
            "Credentials will not be saved due to not providing the "
            "`credentials_path`. To silence this warning pass the "
            "`credentials_path` parameter or set `save_credentials` to False",
            UserWarning
        )

    if gauth.credentials is None:
        gauth.LocalWebserverAuth()
    elif gauth.access_token_expired:
        gauth.Refresh()
    else:
        gauth.Authorize()
    if save_credentials:
        gauth.SaveCredentialsFile(str(credentials_path))

    drive = GoogleDrive(gauth)

    if isinstance(files_paths, (str, Path)):
        files_paths = [files_paths]

    for file_path in files_paths:
        file_path = Path(file_path).resolve()
        file = drive.CreateFile({
            "title": os.path.split(file_path)[-1],
            "parents": [{
                "id": folder_id,
                "kind": "drive#file"
            }]
        })
        file.SetContentFile(str(file_path))
        file.Upload()
        if verbose:
            print(
                "The file: {} of mimeType: {} was succesfully uploaded to "
                "the folder of id: {}."
                .format(file["title"], file["mimeType"], folder_id)
            )
