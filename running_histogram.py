from scipy.io import loadmat
import numpy
from scipy.signal import butter, lfilter, iirfilter, zpk2sos, sosfilt
import matplotlib.pyplot as plt
from pathlib import Path
import rust_acceleration
from event import Event, Site
from obspy import UTCDateTime, Trace
from waveforms import get_stream, InvalidStation, WaveformNotFound
class LowFrequency(Exception):
    def __init__(self, low_frequency, needed_freq):
        pass
def parse_matlab_data(raw_file):
    def error(error_str):
        print(error_str)
        return error_str
    def parse_dictionary(dict):
        output_dictionary = {}
        for key in dict:
            output_dictionary[key] = parse_matlab_data(dict[key])
        return output_dictionary
    def parse_ndarray(array):
        def array_size(array):
            data_size = 1
            for dimension in array.shape:
                data_size = dimension * data_size
            return data_size
        def handle_str_dtype(array):
            if array_size(array) == 1:
                return str(array.flatten()[0])
            else:
                return error(f"handle str dtype with shape: {array.shape}")
        def handle_float64_dtype(array):
            if array_size(array) == 1:
                return array.flatten()[0]
            else:
                return error(f"todo handle f64 array with shape: {array.shape}")
        def handle_int32_dtype(array):

            if len(array.shape) > 1 and array.shape[0]== 1 :
                return handle_int32_dtype(array[0])

            return array
        def handle_object_dtype(array):
            def is_already_flat(array):
                if len(array.shape) > 1:
                    # if len > 1 then every dimension other than the last must have length 1
                    for i in range(len(array.shape) - 1):
                        if array.shape[i] != 1:
                            return False
                    return True
                else:
                    return True
       
          
            if array_size(array) == 1:
                return parse_matlab_data(array.flatten()[0])
            
            elif is_already_flat(array):
                flat_array = array.flatten()
                output_array = []
                for elem in flat_array:
                    output_array.append(parse_matlab_data(elem))
                return output_array

            else:
                return error(f"handle object ndarray with shape {array.shape}")
        def handle_void_dtype(array):
           
            output_dictionary = {}
            for name in array.dtype.names:
 
                output_dictionary[name] = parse_matlab_data(array[name])
            return output_dictionary

        if type(array.dtype) is numpy.dtypes.StrDType:
            return handle_str_dtype(array)
        elif type(array.dtype) is numpy.dtypes.Float64DType:
            return  handle_float64_dtype(array)
        elif type(array.dtype) is numpy.dtypes.ObjectDType:
            return handle_object_dtype(array)
        elif type(array.dtype) is numpy.dtypes.VoidDType:
            return handle_void_dtype(array)
        elif type(array.dtype) == numpy.dtypes.Int32DType:
            return handle_int32_dtype(array)
        else:
            return error(f"INVALID dtype {type(array.dtype)}")
       

    if type(raw_file) is dict:
        return parse_dictionary(raw_file)
    elif type(raw_file) is numpy.ndarray:
        return parse_ndarray(raw_file)
    elif type(raw_file) is str:
        return raw_file
    elif type(raw_file) is bytes:
        return raw_file
    elif type(raw_file) is list:
        output_array = []
        for element in raw_file:
            output_array.append(parse_matlab_data(element))
        return output_array
    else:
        return error(f"INVALID Type: {type(raw_file)}")
def parse_matlab(matlab_file_path):
    raw_data = loadmat("events/event_0219nekhhk.mat", struct_as_record=True)
    return parse_matlab_data(raw_data)
def high_pass(data, sample_rate, low_frequency, corners = 4, zero_phase = False):
    """
    Passes everything below a certian frequency
    uses https://docs.obspy.org/_modules/obspy/signal/filter.html#highpass
    """
    fe = 0.5 * sample_rate
    f = low_frequency / fe
    z, p, k = iirfilter(corners, f, btype='highpass', ftype='butter',
                        output='zpk')
    sos = zpk2sos(z, p, k)
    if zero_phase:
        firstpass = sosfilt(sos, data)
        return sosfilt(sos, firstpass[::-1])[::-1]
    else:
        return sosfilt(sos, data)
    
def bandpass(data, sample_rate, low_frequency, high_frequency):
    # uses https://docs.obspy.org/_modules/obspy/signal/filter.html#bandpass
    if high_frequency >= sample_rate/2.:
        print(f"WARNING upper frequency: {high_frequency} greater then sample_rate/2: {sample_rate/2.}")
        return high_pass(data,sample_rate, low_frequency)
    nyq = 0.5 *  sample_rate
    low = low_frequency / nyq
    high = high_frequency / nyq
    corners = 4
    z, p, k = iirfilter(4, [low, high],btype='band',ftype='butter', output='zpk')
    sos = zpk2sos(z, p, k)
    firstpass = sosfilt(sos, data)
    return sosfilt(sos, firstpass[::-1])[::-1]

def make_histogram(data, number_bins = 10, minimum_value: None | int = None, maximum_value: None | int = None):


    if len(data.shape) != 1:
        print(f"ERROR INVALID SHAPE: {data.shape} should be dimension 1")
   
    
    if maximum_value is None:
        maximum_value = numpy.max(data)
    if minimum_value is None:
        minimum_value = numpy.min(data)
    bins = numpy.linspace(minimum_value, maximum_value, num = number_bins)

    bin_freq = numpy.zeros(number_bins - 1)
    bin_middle = []
    for i in range(0, number_bins-1):
        bin_start = bins[i]
        bin_end = bins[i+ 1]
        number = 0
        for val in data:
            if val >= bin_start and val < bin_end:
                number+=1
        bin_freq[i] = number 

        bin_middle.append((bin_start + bin_end) / 2.0)
    bin_middle = numpy.array(bin_middle)
    return {"bins": bins, "bin_data": bin_freq, "bin_middle": bin_middle}
def histogram_2d(data, x_chunk_size:int, 
                 y_chunks: int, 
                 data_sample_rate: float, arrival_times, start_time: float):
    """
    makes a 2d histogram

    x_chunk_size is a bodge, todo remove dumb code
    """
    hist_list = []
    number_of_x_chunks = int( float(data.shape[0]) / float(x_chunk_size))
    data_min = numpy.min(data)
    data_max = numpy.max(data)

    for i in range(0,  number_of_x_chunks):
        x_start = i * x_chunk_size
        x_end = x_start + x_chunk_size
        data_subset = data[x_start:x_end]
        histogram = make_histogram(
            data_subset, number_bins=y_chunks, 
            minimum_value = data_min,
            maximum_value = data_max
        )
        histogram["start_index"] = x_start
        histogram["final_index"] = x_end
        hist_list.append(histogram)
    arrival_indexes = []
    for time in arrival_times:
        index = (time - start_time) * data_sample_rate / float(x_chunk_size)
        arrival_indexes.append(index)
    return {"histogram_list": hist_list, "arrival_indexes": arrival_indexes}
def rust_histogram_2d(
        data, 
        x_chunk_size:int, 
        y_chunks: int, 
        data_sample_rate: float, 
        arrival_times, start_time: float
):
    rust_output =  rust_acceleration.histogram_2d(
        list(data), x_chunk_size, y_chunks, data_sample_rate, arrival_times, start_time
    )
    output = {'histogram_list': []}
    output["arrival_indexes"] = numpy.array(rust_output["arrival_indices"])
    for histogram in rust_output["histogram_list"]:
        output['histogram_list'].append({
            'bins': numpy.array(histogram['bins']), 
            'bin_data': numpy.array(histogram['bin_data']),
            'bin_middle': numpy.array(histogram['bin_middle'])
        })
        
    return output
def calculate_center_of_mass(histogram_2d):
    line_data = []
    for bin in histogram_2d['histogram_list']:
        sum_mass = 0.
        for m in bin['bin_data']:
            sum_mass += m
        sum_y = 0.
        for y, m in zip(bin['bin_middle'],bin['bin_data']):
            sum_y += y * m
        line_data.append(sum_y / sum_mass)
    return numpy.array(line_data)

def calculate_spread(histogram_2d):
    line_data = []
    for bin in histogram_2d['histogram_list']:
        sum_mass = 0.
        for m in bin['bin_data']:
            sum_mass += m
        sum_y = 0.
        for y, m in zip(bin['bin_middle'],bin['bin_data']):
            sum_y += y * m
        sum_y = sum_y / sum_mass
        output_val = 0.
        for y, m in zip(bin['bin_middle'],bin['bin_data']):
            output_val+= numpy.abs(y - sum_y) * m
        output_val = output_val / sum_mass
        line_data.append(output_val)
    return numpy.array(line_data)
    
def plot_histogram(hist,original_data,):
    fig, [ax1, ax2] = plt.subplots(2, dpi=200)
    ax1.scatter(hist["bin_middle"], hist["bin_data"])
    ax1.set_ylim(bottom = 0)
    ax2.plot(original_data)
    fig.savefig("test_histogram.png")

    plt.clf()
def plot_histogram2d(
        line_data, 
        histogram_2d,
        save_path: Path, 
        aspect_ratio:int = 3, 
        lines = [],
        color_scale_fn = None
):
    def plot_histogram_line(line_data, label,histogram_list,axis, aspect_ratio: float):

        x_data = numpy.linspace(0,len(histogram_list['histogram_list']),len(line_data))

        y_data = line_data * aspect_ratio


        axis.plot(x_data,y_data, label = label)

    
    fig, axis = plt.subplots(figsize=(30.5, 10.5), layout='constrained')

    x_size = len(histogram_2d['histogram_list'])

    y_size = histogram_2d['histogram_list'][0]["bin_data"].shape[0]

    output_image = numpy.zeros((y_size * aspect_ratio,x_size))
    for i, histogram in enumerate(histogram_2d['histogram_list']):
        
        for j, data_point in enumerate(histogram["bin_data"]):
            start_j = (j * aspect_ratio)
            end_j = (j+1) * aspect_ratio
            if color_scale_fn is not None:
                data_point = color_scale_fn(data_point)
            output_image[start_j: end_j,i] = data_point
    axis.invert_yaxis()
    image = axis.imshow(output_image)
    for line in lines:
        plot_histogram_line(line['data'], line['label'],histogram_2d, axis, aspect_ratio)
    if len(lines) >= 1:
        axis.legend()
 
    for time in histogram_2d['arrival_indexes']:
        axis.axvline(time, color="r",linewidth=3., linestyle = '--')
         
    fig.suptitle(f"{save_path}")
    fig.savefig(save_path)
    plt.close()
   

def make_frequency_bins(channel,):
    """
    Returns list of filtered data along side with metadata describing how it was filtered
    """
    high_pass_frequency = 5.0
    middle_bin = [0.1, high_pass_frequency]
    
    output_data = []


    bandpass_data = bandpass(channel['data'],channel['sample_rate'], 
                           middle_bin[0], middle_bin[1])
    output_data.append(
        {"data": bandpass_data, "frequency_bin": middle_bin}
    )

    high_pass_data = high_pass(channel['data'],channel['sample_rate'], high_pass_frequency)
    output_data.append(
        {
            "data": high_pass_data,
            "frequency_bin": [
                high_pass_frequency, 
                channel['sample_rate'] / 2.0]
            }
    )
    output_data.append(
        {
            "data": numpy.copy(channel['data']) - (bandpass_data + high_pass_data),
            "frequency_bin": [
                    0., 
                    middle_bin[0]
                ]
            }
    )

   
    return output_data



class Pick:
    _name: str
    _time: UTCDateTime
    def __init__(self,name:str, time:UTCDateTime):
        self._name = name
        self._time = time
class InputStation:
    _picks: list[Pick]
    _waveforms = None
    _stationName = str
    def __init__(self, 
        picks: list[(UTCDateTime, str)], 
        station_name: str, 
        lead_time_seconds: float = 8.0 * 60. * 60., 
        trail_time_seconds: float = 1.0* 60. * 60.):
        first_time = UTCDateTime("2209-12-31T12:23:34.5")
        self._stationName = station_name
        for time, phase in picks:
            if time < first_time:
                first_time = time
        try:
            self._waveforms = get_stream(station_name, None,first_time - lead_time_seconds, first_time + trail_time_seconds)
        except InvalidStation  as e:
            print("skipping station station_name as it is invalid")
        except WaveformNotFound as e:
            print(e)
        self._picks = list(map(lambda x: Pick(x[1],x[0]),picks))
        pass
    def picks(self) -> list[Pick]:
        return self.picks
    def waveforms(self):
        return self._waveforms
    def stationName(self):
        return self._stationName

class HistogramInput:
    _stations: list[InputStation]
    def __init__(self, event: Event, max_distance = None):
        """
        Max distance is in meters
        """
        sites = {}
        for pick in event.picks():
            print(pick.site().stationName())
            print(pick.phase())
            print(pick.time())
            distance = event.distance_from_site(pick.site())
            print(f"distance: {distance}")
            if max_distance is not None:
                if distance > max_distance:
                    continue
            
            if sites.get(pick.site().stationName()) is None:
                sites[pick.site().stationName()] = {"site": pick.site(), "picks": []}
            sites[pick.site().stationName()] ["picks"].append(pick)
        self._stations = []
        print(sites)
        for site_name, site in sites.items():
            input_station_picks = []
            print(site)
            for pick in site["picks"]:
                
                input_station_picks.append((pick.time(), pick.phase()))
            self._stations.append(
                InputStation(input_station_picks,site["site"].stationName())
            )
     

    def stations(self) -> list[InputStation]:
        return self._stations
def trace_to_channel(trace: Trace):
    return {
        'data': trace.data, 
        'channel': trace.stats['channel'],
        'sample_rate': trace.stats['sampling_rate'],
        'start_time': trace.stats['starttime'],
        'end_time': trace.stats['endtime']
    }

def save_station(input: HistogramInput):
    skip_channels =  ['BNE', 'BNN', 'BNZ', "LHE", "LHN", "LHZ", "LCE", 'VM6','LCQ', "LDI"]
    for station in input.stations():
        for trace in station.waveforms():
            print(trace)
            print(trace.stats)
            channel = trace_to_channel(trace)
            if trace.stats['channel'] in skip_channels:
                print(f"skipping channel \"{trace.stats['channel']}\" ")
                continue


            bins = make_frequency_bins(channel)
            for bin in bins:
                # first the rust version
                scaled_data = numpy.log(numpy.abs(bin['data']) + 1)
                print(scaled_data)
                
                hist2d = rust_histogram_2d(scaled_data,10000, 100, channel['sample_rate'],[],channel['start_time'])
    
                center_of_mass = calculate_center_of_mass(hist2d)
                spread = calculate_spread(hist2d)
                frequency_str = ""
                if len(bin["frequency_bin"]) == 2:
                    frequency_str = f"{bin["frequency_bin"][0]}_{bin["frequency_bin"][1]}"
                root_path = Path(f"rust_figures/{station.stationName()}/{channel["channel"]}")
                root_path.mkdir(exist_ok=True, parents = True)
                write_path = root_path / Path(f"{frequency_str}.png")

                #color_scale = lambda x: numpy.log(x + 1.)
                color_scale = lambda x: x
                lines = [
                    {"data": center_of_mass, "label": "center of mass"},
                    {"data": spread, "label": "spread"}
                ]

                plot_histogram2d(
                    channel['data'],
                    hist2d, 
                    write_path,

                    lines = lines,
                    color_scale_fn = color_scale
                )


        pass
def histogram_from_event(event: Event, max_distance = None):
    print(f"todo process event: {event}")
    print(f"num picks: {len(event.picks())}")
    for pick in event.picks():
        print(f"\tpics: {pick.site().stationName()}")
    histogram_input = HistogramInput(event, max_distance)
    save_station(histogram_input)
    
    pass
