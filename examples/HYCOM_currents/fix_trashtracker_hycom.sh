# usage:
# ./fix_trashtracker_hycom.sh "wildcard_path_to_u_files" "wildcard_path_to_v_files"
#     note: quotes required around wildcard paths (e.g. "./dir/*.nc")

# A netcdf file which "packs" its float32 values into int16s with a scaling, offset, and fill value, should have a set
#   of standard attributes which specify these things; these attribute names are defined by netcdf convention,
#   and their presence allows libraries reading in netcdf files to automatically unpack the values.
# However, any hycom files downloaded by scripts in the trashtracker repo do not contain these attributes; they are
#   lost during the concatenation phase, since it is done manually rather than using a purpose-built tool such as ncrcat.
# This script modifies the files in-place, so be careful.  Make a copy if you want to be safe.
# To understand the ncatted command check out https://linux.die.net/man/1/ncatted

if [ "$#" -ne 2 ]; then
  echo "Error: script requires exactly two arguments."
  exit 1;
fi

FILL_VALUE=-30000
SCALE_FACTOR=0.001

for f in $1
do
    echo "Processing $f file.."
    ncatted -a _FillValue,water_u,c,s,$FILL_VALUE -a scale_factor,water_u,c,f,$SCALE_FACTOR $f
done

for f in $2
do
    echo "Processing $f file..."
    ncatted -a _FillValue,water_v,c,s,$FILL_VALUE -a scale_factor,water_v,c,f,$SCALE_FACTOR $f
done
