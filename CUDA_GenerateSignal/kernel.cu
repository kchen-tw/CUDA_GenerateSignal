#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <iomanip> s

// 包含CUDA頭文件
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuComplex.h>

using namespace std;

// 定義複數型別
typedef cuDoubleComplex Complex;

// 自定義 CUDA專用的 π 常數
__device__ const double PI = 3.14159265358979323846;

// 在GPU上生成正弦波
// waveform: 用於存儲正弦波數據的陣列
// N: 波形的總點數
// frequency: 正弦波的頻率（以赫茲為單位）
// amplitude: 正弦波的振幅
// samplingRate: 采樣率（以赫茲為單位）
__global__
void generateSineWave(Complex* waveform, int N, double* frequencies, double* amplitudes, int numFrequencies, double samplingRate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;


    if (idx < N) {
        double t = static_cast<double>(idx) / samplingRate;  // 計算時間

        // 生成正弦波
        for (int i = 0; i < numFrequencies; ++i) {
            double omega = 2.0 * PI * frequencies[i] * t;
            waveform[idx] = cuCadd(waveform[idx], make_cuDoubleComplex(amplitudes[i] * cos(omega), amplitudes[i] * sin(omega)));
        }
    }
}

int main() {
    const int N = 1024; // 波形點數
    const double samplingRate = 44100.0; // 例：取樣率為 44100 Hz


    // 宣告Host端的記憶體
    vector<Complex> h_waveform(N);
    // C4=261.63 (Do), D4=293.67 (Re), E4=329.64(Mi), F4=349.24(Fa), G4=392(So), A4=440(La), B4=493.88(Si)
    vector<double> h_frequencies = { 440.0, 261.63 };  // 頻率
    vector<double> h_amplitudes = { 1.0, 0.5 };  // 振幅

    // Device端的記憶體
    Complex* d_waveform;
    double* d_frequencies;
    double* d_amplitudes;

    cudaMalloc((void**)&d_waveform, N * sizeof(Complex));
    cudaMalloc((void**)&d_frequencies, h_frequencies.size() * sizeof(double));
    cudaMalloc((void**)&d_amplitudes, h_amplitudes.size() * sizeof(double));

    // 將頻率和振幅從Host複製到Device
    cudaMemcpy(d_frequencies, h_frequencies.data(), h_frequencies.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_amplitudes, h_amplitudes.data(), h_amplitudes.size() * sizeof(double), cudaMemcpyHostToDevice);

     // 使用 cudaOccupancyMaxPotentialBlockSize 動態估算最佳的線程塊大小
    int threadsPerBlock, minGridSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &threadsPerBlock, generateSineWave, 0, 0);

    // 計算 gridSize
    int gridSize = (N + threadsPerBlock - 1) / threadsPerBlock;

    // 在GPU上執行生成正弦波的函數
    generateSineWave << <gridSize, threadsPerBlock >> > (d_waveform, N, d_frequencies, d_amplitudes, h_frequencies.size(), samplingRate);


    // 將結果從Device端複製到Host端
    cudaMemcpy(h_waveform.data(), d_waveform, N * sizeof(Complex), cudaMemcpyDeviceToHost);

    // 將正弦波寫入檔案
    ofstream waveformFile("sine_waveform.txt");
    waveformFile << fixed << setprecision(8); // 設定小數點後8位
    for (const auto& value : h_waveform) {
        waveformFile << cuCreal(value) << " " << cuCimag(value) << endl;
    }
    waveformFile.close();
    cout << "Sine wave file 'sine_waveform.txt' created successfully." << endl;


    // 釋放Device端的記憶體
    cudaFree(d_waveform);
    cudaFree(d_frequencies);
    cudaFree(d_amplitudes);

    // 重置 CUDA 裝置狀態
    cudaDeviceReset();

    return 0;
}