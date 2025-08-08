template<typename T>
struct Vector {


};

template<typename T>
class Matrix {

};

template<typename T>
using Vec = std::vector<T>;

static_assert(sizeof(int) == 4, "int must be 4 bytes");
static_assert(sizeof(float) == 4, "float must be 4 bytes");

constexpr float PI = 3.14159;

constexpr int MAX_SIZE = 1000;

constexpr int CUDA_BLOCK_SIZE = 256;

constexpr float CUDA_CONSTANTS = 1.0;





