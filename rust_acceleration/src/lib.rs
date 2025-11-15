use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
mod rust_acceleration {
    use pyo3::{exceptions::PyValueError, prelude::*};
    #[derive(IntoPyObject)]
    pub struct PyHistogram2d {
        pub histogram_list: Vec<PyHistogram>,
        pub arrival_indices: Vec<f64>,
    }
    #[derive(IntoPyObject)]
    pub struct PyHistogram {
        pub bins: Vec<f32>,
        pub bin_data: Vec<f32>,
        pub bin_middle: Vec<f32>,
    }
    fn histogram_1d(
        data: &[f32],
        number_of_bins: u32,
        minimum_value: f32,
        maximum_value: f32,
    ) -> PyHistogram {
        let bins = (0..number_of_bins)
            .map(|i| i as f32 * (maximum_value - minimum_value) / number_of_bins as f32)
            .collect::<Vec<_>>();
        let mut bin_data = Vec::new();
        let mut bin_middle = Vec::new();
        for i in 0..(bins.len() - 1) {
            let bin_start = bins[i];
            let bin_end = bins[i + 1];
            let sum: f32 = data
                .iter()
                .filter(|value| **value >= bin_start && **value < bin_end)
                .sum();
            bin_data.push(sum);
            bin_middle.push((bin_start + bin_end) / 2.0);
        }
        PyHistogram {
            bins,
            bin_data,
            bin_middle,
        }
    }

    /// Formats the sum of two numbers as string.
    #[pyfunction]
    fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
        Ok((a + b).to_string())
    }

    #[pyfunction]
    pub fn histogram_2d(
        data: Vec<f32>,
        x_chunk_size: usize,
        y_chunks: u32,
        data_sample_rate: f64,
        arrival_times: Vec<f64>,
        start_time: f64,
    ) -> PyResult<PyHistogram2d> {
        println!("data.len(): {:#?}", data.len());
        println!("x chunk size: {:#?}", x_chunk_size);
        println!("y_chunks: {:#?}", y_chunks);
        println!("data sample_rate: {:#?}", data_sample_rate);
        println!("arrival_times: {:#?}", arrival_times);
        println!("start_time: {:#?}", start_time);

        if data.is_empty() {
            return Err(PyValueError::new_err("data has length zero"));
        }
        let number_x_chunks = data.len() / x_chunk_size as usize;
        let data_min = *data
            .iter()
            .min_by(|a, b| a.total_cmp(b))
            .expect("array length should be zero");
        let data_max = *data
            .iter()
            .max_by(|a, b| a.total_cmp(b))
            .expect("array length should be zero");
        let histogram_list = (0..number_x_chunks)
            .map(|i| {
                let x_start = i * x_chunk_size;
                let x_end = x_start + x_chunk_size;
                let data_subset = &data[x_start..x_end];
                histogram_1d(data_subset, y_chunks, data_min, data_max)
            })
            .collect();
        let arrival_indices = arrival_times
            .iter()
            .map(|time| (time - start_time) * data_sample_rate / (x_chunk_size as f64))
            .collect();

        Ok(PyHistogram2d {
            histogram_list,
            arrival_indices,
        })
    }
}
