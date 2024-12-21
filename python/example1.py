import xarray as xr
import fsspec
import ujson
from kerchunk.combine import MultiZarrToZarr
from kerchunk.grib2 import scan_grib
import concurrent.futures

# Define constants for configuration
STORAGE_OPTIONS = {"anon": True}
FILTER_CONDITIONS = {'typeOfLevel': 'heightAboveGround', 'level': [2, 10]}
JSON_DIR = '/tmp/jsons/'

# Initialize file systems
fs_read = fsspec.filesystem('s3', anon=True, skip_instance_cache=True)
# fs_write = fsspec.filesystem('s3', anon=False)

USE_S3_WRITE = False

# Update file systems accordingly
if USE_S3_WRITE:
    fs_write = fsspec.filesystem('s3', anon=False)
else:
    fs_write = fsspec.filesystem('file')  # Local filesystem



def make_json_name(file_url: str, message_number: int) -> str:
    """Create a unique name for each reference JSON file."""
    date_segment = file_url.split('/')[3].split('.')[1]
    name_segments = file_url.split('/')[5].split('.')[1:3]
    return f'{JSON_DIR}{date_segment}_{name_segments[0]}_{name_segments[1]}_message{message_number}.json'


def generate_json_files(file_url: str) -> None:
    """Generate JSON references from a GRIB2 file."""
    print(f'Generating JSON references for {file_url}...')
    references = scan_grib(file_url, storage_options=STORAGE_OPTIONS, filter=FILTER_CONDITIONS)
    for i, message in enumerate(references):
        json_file_name = make_json_name(file_url, i)
        print(f'Writing {json_file_name}...')
        with fs_write.open(json_file_name, "w") as f:
            f.write(ujson.dumps(message))


def process_files(file_list: list[str]) -> None:
    """Process all GRIB2 files to generate JSON files."""
    for file_url in file_list:
        generate_json_files(file_url)

def process_files_parallel(file_list: list[str]) -> None:
    """Process all GRIB2 files to generate JSON files in parallel."""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(generate_json_files, file_list)



def find_grib_files() -> list[str]:
    """Locate and return a list of GRIB2 files."""

    # valid path looks like this: s3://noaa-hrrr-bdp-pds/hrrr.20140730/conus/hrrr.t18z.wrfsfcf01.grib2
    available_days = fs_read.glob('s3://noaa-hrrr-bdp-pds/hrrr.*')
    print(available_days)
    # an available day looks like this: 'noaa-hrrr-bdp-pds/hrrr.20241210'
    MAX_DAYS = 2
    if len(available_days) > MAX_DAYS:
        available_days = available_days[:MAX_DAYS]
    file_paths = [
        # f's3://{day}/conus/{file}'
        f's3://{file}'
        for day in available_days
        for file in fs_read.glob(f's3://{day}/conus/*wrfsfcf01.grib2')
    ]
    return sorted(file_paths)


def combine_references(json_dir: str) -> xr.Dataset:
    """Combine referenced JSON files into a single xarray Dataset."""
    reference_jsons = sorted(fs_write.ls(json_dir))
    mzz = MultiZarrToZarr(
        reference_jsons,
        concat_dims=['valid_time'],
        identical_dims=['latitude', 'longitude', 'heightAboveGround', 'step']
    )
    translated_data = mzz.translate()

    # Use fsspec to load the dataset
    fs = fsspec.filesystem("reference", fo=translated_data, remote_protocol='s3', remote_options=STORAGE_OPTIONS)
    mapper = fs.get_mapper("")
    return xr.open_dataset(mapper, engine="zarr", backend_kwargs=dict(consolidated=False), chunks={'valid_time': 1})


def main():
    print("Finding GRIB2 files...")
    grib_files = find_grib_files()

    print("Generating JSON files...")
    # process_files(grib_files)
    process_files_parallel(grib_files)

    print("Combining references into xarray Dataset...")
    dataset = combine_references(JSON_DIR)

    print("Plotting results...")
    import matplotlib.pyplot as plt
    plot = dataset['d2m'][-1].plot()
    plt.savefig("d2m_plot.png")


if __name__ == "__main__":
    main()
