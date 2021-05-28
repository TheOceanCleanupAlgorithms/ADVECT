from getpass import getpass


def get_ECCO_credentials():
    print(
        "Download requires authentication. "
        "You can find your WebDAV credenials and/or "
        "create an account here: https://ecco.jpl.nasa.gov/drive/"
    )
    user = input("Enter your WebDAV username: ")
    password = getpass("Enter your WebDAV password: ")
    return user, password
