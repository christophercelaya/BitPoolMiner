# BitPool Miner
 GPU (CUDA) Miner for [BitClout](https://bitclout.com).  The miner works with the pool [BitPool.me](https://bitpool.me).

### Prerequisites

1. [CUDA Toolkit 11.2.2](https://developer.nvidia.com/cuda-11.2.2-download-archive)
2. [CMake](https://cmake.org)
3. [Socket.IO client](https://github.com/socketio/socket.io-client-cpp/blob/master/INSTALL.md#with-cmake)

### Installation

1. Clone the repo
    ```sh
    git clone https://github.com/lobovkin/BitPoolMiner.git
    ```
2. Build the BitPool Miner
    ```sh
    cd BitPoolMiner
    cmake .
    make
    ```
3. Run the BitPool Miner
    ```sh
    ./bitpoolminer --address YOUR_PUBLIC_KEY --worker ANY_WORKER_NAME
    ```

## License

Distributed under the GPLv3 License. See `LICENSE` for more information.

## Contact

Anton Lobovkin - [@lobovin](https://bitclout.com/u/lobovkin)  
[Discord Channel](https://discord.com/channels/820740896181452841/844318509777420298)
