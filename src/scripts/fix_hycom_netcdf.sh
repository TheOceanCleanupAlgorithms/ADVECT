# Axel: to understand this command, simply check out https://linux.die.net/man/1/ncatted
# I copied some data from the storage2 to this dir for testing.
# before/after running do ncinfo -v water_u <u_path> and ncinfo water_v <v_path> to see what it's doing.
# it modifies the files in-place so to try again, just rm the <hycom_file> and cp the <hycom_file>_orig that I included in this dir to reset the file.
# to run this on all the hycom data, just replace the U_PATH and V_PATH with wildcards to the hycom files.  even though it's just modifying attributes inplace, it seems to take a long time.  sorry...
FILL_VALUE=-30000
SCALE_FACTOR=0.001
U_PATH="./u_1993*.nc"
V_PATH="./v_1993*.nc"
ncatted -a _FillValue,water_u,c,s,$FILL_VALUE -a scale_factor,water_u,c,f,$SCALE_FACTOR $U_PATH
ncatted -a _FillValue,water_v,c,s,$FILL_VALUE -a scale_factor,water_v,c,f,$SCALE_FACTOR $V_PATH
