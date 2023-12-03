#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>

// 包含CUDA頭文件
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


// 自定義 CUDA專用的 π 常數
__device__ const double PI = 3.14159265358979323846;

// 在GPU上生成正弦波
// waveform: 用於存儲正弦波數據的陣列
// N: 波形的總點數
// frequency: 正弦波的頻率（以赫茲為單位）
// amplitude: 正弦波的振幅
// samplingRate: 采樣率（以赫茲為單位）
__global__
void generateSineWave(double* waveform, int N, double frequency, double amplitude, double samplingRate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        // 計算時間（單位：秒）
        double t = idx / samplingRate;

        // 計算正弦波的數值並存儲在waveform陣列中
        waveform[idx] = amplitude * sin(2.0 * PI * frequency * t);

    }
}

int main() {
    const int N = 1024; // 波形點數
    const double samplingRate = 44100.0; // 例：取樣率為 44100 Hz
    // C4=261.63 (Do), D4=293.67 (Re), E4=329.64(Mi), F4=349.24(Fa), G4=392(So), A4=440(La), B4=493.88(Si)
    const double frequency = 440; // 設定頻率 
    const double amplitude = 1;  // 振幅

    // 宣告Host端的記憶體
    std::vector<double> h_waveform(N);

    // 宣告Device端的記憶體
    double* d_waveform;
    cudaMalloc((void**)&d_waveform, N * sizeof(double));

     // 使用 cudaOccupancyMaxPotentialBlockSize 動態估算最佳的線程塊大小
    int threadsPerBlock, minGridSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &threadsPerBlock, generateSineWave, 0, 0);

    // 計算 gridSize
    int gridSize = (N + threadsPerBlock - 1) / threadsPerBlock;

    // 在GPU上執行生成正弦波的函數
    generateSineWave << <gridSize, threadsPerBlock >> > (d_waveform, N, frequency, amplitude, samplingRate);

    // 將結果從Device端複製到Host端
    cudaMemcpy(h_waveform.data(), d_waveform, N * sizeof(double), cudaMemcpyDeviceToHost);

    // 釋放Device端的記憶體
    cudaFree(d_waveform);

    // 重置 CUDA 裝置狀態
    cudaDeviceReset();

    // 將波形寫入檔案
    std::ofstream waveformFile("sine_wave_cuda.txt");
    for (const auto& value : h_waveform) {
        waveformFile << value << "\n";
    }
    waveformFile.close();

    std::cout << "Sine wave file 'sine_wave_cuda.txt' created successfully." << std::endl;

    return 0;
}