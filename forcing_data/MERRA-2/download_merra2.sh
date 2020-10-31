# unfortunately the url I used are temporary.
# you can do data selection/processing and get your own collection of urls here:
#  https://disc.sci.gsfc.nasa.gov/datasets/M2I6NPANA_5.12.4/summary
# also, downloading directions at that site.  requires some setup.

txtfile="subset_M2I6NPANA_5.12.4_20201031_010827.txt"
wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --auth-no-challenge=on --keep-session-cookies --content-disposition -i $txtfile -P ./SURFACE_WIND/
