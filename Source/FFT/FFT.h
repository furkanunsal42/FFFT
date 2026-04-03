#pragma once

#include "Texture1D.h"
#include "Texture2D.h"
#include "Texture2DArray.h"
#include "Texture3D.h"
#include "Buffer.h"

class FFFT {
public:

	enum fft_dimension {
		x,
		y,
		z,
		xy,
		yz,
		xz,
		xyz,
	};

	template<typename T>
	static constexpr fft_dimension default_fft_dimension();

	enum windowing_function {
		hann,
		hamming,
	};

	enum fft_algorithm {
		fft_dft,
		fft_radix2_zeropadding,
		fft_radix2,
		fft_radix235_zeropadding,
		fft_radix235,
	};

	enum component {
		real,
		complex,
		real_complex,
	};

	template<typename T> void fft		(T&	source,	T& target, fft_dimension dimension = default_fft_dimension<T>(), fft_algorithm algorithm = radix235_fft, glm::ivec3 offset = glm::ivec3(0), glm::ivec3 size = glm::ivec3(0));
	template<typename T> void i_fft		(T&	source,	T& target, fft_dimension dimension = default_fft_dimension<T>(), fft_algorithm algorithm = radix235_fft, glm::ivec3 offset = glm::ivec3(0), glm::ivec3 size = glm::ivec3(0));
	
	template<typename T> void shift		(T&	source,	T& target, glm::ivec3 shift_size, fft_dimension dimension = default_fft_dimension<T>(), glm::ivec3 offset = glm::ivec3(0), glm::ivec3 size = glm::ivec3(0));
	template<typename T> void i_shift	(T& source, T& target, glm::ivec3 shift_size, fft_dimension dimension = default_fft_dimension<T>(), glm::ivec3 offset = glm::ivec3(0), glm::ivec3 size = glm::ivec3(0));

	template<typename T> void copy		(T& source,	T& target, component comp = real_complex, glm::ivec3 offset = glm::ivec3(0), glm::ivec3 size = glm::ivec3(0));
	
	template<typename T> std::shared_ptr<T> create	(T& source, component comp = real_complex, glm::ivec3 offset = glm::ivec3(0), glm::ivec3 size = glm::ivec3(0));
	template<typename T> std::shared_ptr<T> pad		(T& source, T& target, component comp = real_complex, glm::ivec3 offset = glm::ivec3(0), glm::ivec3 size = glm::ivec3(0));
	template<typename T> std::shared_ptr<T> i_pad	(T& source, T& target, component comp = real_complex, glm::ivec3 offset = glm::ivec3(0), glm::ivec3 size = glm::ivec3(0));

	//void window();
	//void inverse_window();
	//void conjugate();
	//void multiply();
	//void divide();
	//void add();
	//void subtract();

	template<typename T> bool is_complex(T& source);
	template<typename T> bool is_real(T& source);

private:


	std::vector<std::pair<std::string, std::string>> generate_shader_macros();
	void compile_shaders();

	template<typename T> void dft		(T& source, T& target, fft_dimension dimension = default_fft_dimension<T>(), glm::ivec3 offset = glm::ivec3(0), glm::ivec3 size = glm::ivec3(0));
	template<typename T> void radix2	(T& source, T& target, fft_dimension dimension = default_fft_dimension<T>(), glm::ivec3 offset = glm::ivec3(0), glm::ivec3 size = glm::ivec3(0));
	template<typename T> void radix3	(T& source, T& target, fft_dimension dimension = default_fft_dimension<T>(), glm::ivec3 offset = glm::ivec3(0), glm::ivec3 size = glm::ivec3(0));
	template<typename T> void radix5	(T& source, T& target, fft_dimension dimension = default_fft_dimension<T>(), glm::ivec3 offset = glm::ivec3(0), glm::ivec3 size = glm::ivec3(0));
	
	struct kernels {
		std::shared_ptr<ComputeProgram> cp_dft;
	};

	std::unordered_map<size_t, kernels> type_to_kernels_table;
};

#include "FFT_Templated.h"
