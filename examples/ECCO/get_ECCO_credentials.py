import tempfile
from getpass import getpass
import subprocess


def get_ECCO_credentials():
    print(
        "Download requires authentication. "
        "You can find your WebDAV credenials and/or "
        "create an account here: https://ecco.jpl.nasa.gov/drive/"
    )
    test_url = "https://ecco.jpl.nasa.gov/drive/files/README"
    while True:
        user = input("Enter your WebDAV username: ")
        password = getpass("Enter your WebDAV password: ")
        with tempfile.TemporaryDirectory() as tmp_dir:
            wget_command = (
                f"wget --user {user} --password {password} -P {tmp_dir} {test_url}"
            )
            response = subprocess.getoutput(wget_command)

        if "200 OK" in response:
            print("Authentication Successful.")
            return user, password
        elif response.count("401 Unauthorized") == 2:
            # always 1 count because of the way the server handles
            # the request.  Second count means the user/pass is bad.
            print("Authentication failed.  Try again.")
            print("Find your WebDAV credenials here: https://ecco.jpl.nasa.gov/drive/")
        else:
            raise RuntimeError(
                "Unexpected failure while authenticating credentials.  Aborting.\n"
                f"Test wget command: '{wget_command}'\n"
                f"Output from test wget command:\n '{response}'"
            )
