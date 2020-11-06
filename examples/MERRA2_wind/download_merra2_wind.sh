# this command requires some additional setup, namely you have to store your earthdata username and
# password in a local cookie file.  An account is free to set up.  Further directions can be found
# here: https://disc.gsfc.nasa.gov/data-access

# the url list txtfile may be temporary, so this may fail in the future.
# If so, you can do data selection/processing and get your own collection of urls here:
#  https://disc.sci.gsfc.nasa.gov/datasets/M2I6NPANA_5.12.4/summary

txtfile="merra2_urls.txt"
wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --auth-no-challenge=on --keep-session-cookies --content-disposition -i $txtfile -P ./
