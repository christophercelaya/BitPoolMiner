#include <iostream>
#include <sio_client.h>
#include <sio_message.h>
#include <pthread.h>
#include <chrono>
#include <thread>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>
#include <atomic>
#include <ctime>
#include <iomanip>
#include <cmath>

#define MAX_GPUS 24
#define NONCES 3
sio::client io;
static uint32_t work[25] = {0};

extern void clouthash_init(uint8_t id);
extern void clouthash_run(uint8_t id, uint32_t blocks, uint32_t *nonces);
extern void clouthash_update(uint64_t *data);

void miner(uint8_t id, char * name, uint32_t blocks) {
  using namespace std::chrono_literals;
  cudaSetDevice(id);
  clouthash_init(id);
  uint32_t incr = 0;
  uint32_t nonces[10];
  uint32_t data[50] = {0};
  std::time_t time;
  struct std::tm *localtime;
  std::chrono::high_resolution_clock::time_point times[10];
  int times_index = 1;
  int times_count = 1;
  data[25] = 0x00000006;
  data[33] = 0x80000000;
  times[0] = std::chrono::high_resolution_clock::now();
  while (1) {
    if (work[19] == 0 && work[20] == 0) {
      std::this_thread::sleep_for(100ms);
      continue;
    }
    if (times_count == 1) {
      time = std::time(nullptr);
      localtime = std::localtime(&time);
      std::cout<<"["<<std::put_time(localtime, "%Y-%m-%d %H:%M:%S")<<"] [hashrate] GPU #"<<+id<<" "<<name<<" "<<"Benchmark..."<<std::endl;
    }
    for (int i = 0; i < 23; i++) data[i] = work[i];
    data[23] = incr << 8 | id;
    if (incr == 0xffffff) incr = 0; else incr++;
    cudaSetDevice(id);
    clouthash_update((uint64_t*)data);
    clouthash_run(id, blocks, nonces);
    times[times_index] = std::chrono::high_resolution_clock::now();
    if (times_count < 10) times_count++;
    std::chrono::duration<double, std::micro> duration = times[times_index] - times[times_index + 1 == times_count ? 0 : times_index + 1];
    if (times_index == 9) times_index = 0; else times_index++;
    double hashrate = (double)blocks * 1024.0 * (times_count - 1) / duration.count();
    time = std::time(nullptr);
    localtime = std::localtime(&time);
    if (times_count > 2 && times_index % 2 == 0) std::cout<<"["<<std::put_time(localtime, "%Y-%m-%d %H:%M:%S")<<"] [hashrate] GPU #"<<+id<<" "<<name<<" "<<hashrate<<" MH/s"<<std::endl;
    for (int i = 0; i < NONCES; i++) {
      if (nonces[i] != 0xffffffff) {
        data[24] = nonces[i];
        std::cout<<"["<<std::put_time(localtime, "%Y-%m-%d %H:%M:%S")<<"] [share] Found!"<<std::endl;
        io.socket()->emit("share", std::make_shared<std::string>((char *)data, 100));
      } else break;
    }

  }
}

int main(int argc, char *argv[]) {
  if (argc < 4) {
    std::cerr<<"Usage: "<<argv[0]<<" --address YOUR_PUBLIC_KEY --worker ANY_WORKER_NAME"<<std::endl;
    return 1;
  }
  std::map<std::string, std::string> query;
  for (int i = 1; i < argc - 1; i++) {
    if (std::string(argv[i]) == "--address") query["address"] = argv[++i];
    else if (std::string(argv[i]) == "--worker") query["worker"] = argv[++i];
  }
  static int devices = 0;
  static char buf[100];
  cudaError_t err = cudaGetDeviceCount(&devices);
  if (err != cudaSuccess) {
    std::cout<<"Can't get CUDA devices"<<std::endl;
    return 1;
  }
  if (devices > MAX_GPUS) devices = MAX_GPUS;
  static std::vector<std::thread> threads;
  cudaDeviceProp deviceProp;
  uint32_t blocks;
  for (uint8_t i = 0; i < devices; ++i) {
    cudaGetDeviceProperties(&deviceProp, i);
    if (deviceProp.major >= 5) {
      std::cout<<"#"<<+i<<" "<<deviceProp.name<<" "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
      if (deviceProp.maxGridSize[0] < 2048 * 1024) blocks = deviceProp.maxGridSize[0]; else blocks = 2048 * 1024;
      cudaSetDevice(i);
      cudaDeviceReset();
      cudaSetDeviceFlags(cudaDeviceScheduleYield);
      threads.push_back(std::thread(miner, i, deviceProp.name, blocks));
    } else {
      std::cout<<"#"<<+i<<" "<<deviceProp.name<<" "<<deviceProp.major<<"."<<deviceProp.minor<<" is too old"<<std::endl;
    }
  }
  if (threads.size() == 0) return 1;
  std::cout<<"*** BitPool.me Miner v1.0.0 started on "<<threads.size()<<" GPU"<<(threads.size() == 1 ? "" : "s")<<" ***"<<std::endl;
  io.connect("http://v1.bitpool.me", query);
  io.socket()->on("work", [&](sio::event& ev) {
    std::string data = ev.get_message()->get_string();
    for (int i = 0; i < 100; i++) {
      buf[i] = strtoll(data.substr(i * 2, 2).c_str(), NULL, 16);
    }
    memcpy(work, buf, 100);
  });
  io.socket()->on("error", [&](sio::event& ev) {
    std::string text = ev.get_message()->get_string();
    static time_t time = std::time(nullptr);;
    static struct tm *localtime = std::localtime(&time);;
    std::cout<<"["<<std::put_time(localtime, "%Y-%m-%d %H:%M:%S")<<"] [share] "<<text<<std::endl;
  });
  io.socket()->on("accepted", [&](sio::event& ev) {
    std::string text = ev.get_message()->get_string();
    static time_t time = std::time(nullptr);;
    static struct tm *localtime = std::localtime(&time);;
    std::cout<<"["<<std::put_time(localtime, "%Y-%m-%d %H:%M:%S")<<"] [share] "<<text<<std::endl;
  });
  for (std::thread & thread : threads) {
    thread.join();
  }
  return 0;
}
