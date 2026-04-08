#pragma once

#include "Texture1D.h"
#include "Texture2D.h"
#include "Texture2DArray.h"
#include "Texture3D.h"
#include "Buffer.h"

#include "ComputeProgram.h"
#include "Tools/VariantedComputeProgram.h"

#include <filesystem>

namespace shader_directory {
	static std::filesystem::path ffft2_shader_directory = "../FFFT/Source/GLSL/FFT/";
}

class FFFT2 {
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
	constexpr static fft_dimension default_fft_dimension();

	enum windowing_function {
		hann,
		hamming,
	};

	enum fft_algorithm {
		fft_dft,
		fft_radix2_zeropadding,
		fft_radix2_dft,
		fft_radix235_zeropadding,
		fft_radix235_dft,
	};

	enum component {
		real,
		complex,
		real_complex,
	};

	template<typename T> void				fft		(T&	source,	T& target, fft_dimension dimension = default_fft_dimension<T>(), fft_algorithm algorithm = fft_radix235_dft);
	template<typename T> void				i_fft	(T&	source,	T& target, fft_dimension dimension = default_fft_dimension<T>(), fft_algorithm algorithm = fft_radix235_dft);
	
	template<typename T> void				shift	(T&	source,	T& target, glm::ivec3 shift_size);
	template<typename T> std::shared_ptr<T>	shift	(T&	source, glm::ivec3 shift_size);
	template<typename T> void				i_shift	(T& source, T& target, glm::ivec3 shift_size);
	template<typename T> std::shared_ptr<T>	i_shift	(T& source, glm::ivec3 shift_size);

	template<typename T> void				copy	(T& source,	T& target, component comp = real_complex, glm::ivec3 source_offset = glm::ivec3(0), glm::ivec3 target_offset = glm::ivec3(0), glm::ivec3 size = glm::ivec3(0));
	template<typename T> std::shared_ptr<T> create	(T& source, component comp = real_complex, glm::ivec3 size = glm::ivec3(0));

	template<typename T> void				pad		(T& source, T& target, glm::ivec3 offset = glm::ivec3(0), glm::vec2 padding_value = glm::vec2(0));
	template<typename T> void				i_pad	(T& source, T& target, glm::ivec3 offset = glm::ivec3(0));
	template<typename T> std::shared_ptr<T> pad		(T& source, glm::ivec3 padded_size, glm::ivec3 offset = glm::ivec3(0), glm::vec2 padding_value = glm::vec2(0));
	template<typename T> std::shared_ptr<T> i_pad	(T& source, glm::ivec3 offset, glm::ivec3 size);

	//void window();
	//void inverse_window();
	//void conjugate();
	//void multiply();
	//void divide();
	//void add();
	//void subtract();

//private:

	template<typename T> static bool is_complex(T& source);
	template<typename T> static bool is_real(T& source);
	template<typename T> static bool is_same(T& source, T& target);

	std::string component_to_string(component component);

	Texture2D::ColorTextureFormat complex_texture_format(Texture2D::ColorTextureFormat real_texture);
	Texture2D::ColorTextureFormat real_texture_format(Texture2D::ColorTextureFormat complex_texture);

	void compile_shaders();

	struct fft_iteration {
		constexpr static size_t radix_dft = 1;
		size_t	radix = radix_dft;
		size_t	chunk_size = 0;
	};

	struct fft_plan {
		std::vector<fft_iteration> iterations;
	};

	fft_plan create_plan(size_t array_size, size_t supported_max_radix, size_t supported_min_radix = fft_iteration::radix_dft);
	fft_plan create_plan(size_t array_size, const std::vector<size_t>& supported_radixes = { 25, 16, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2 });
	
	template<typename T> void split		(T& source, T& target, glm::ivec3 group_count);
	template<typename T> void step		(T& source, T& target, size_t radix, fft_dimension dimension = default_fft_dimension<T>(), bool inverse = false);

	bool shaders_are_set = false;

	VariantedComputeProgram cp_copy;
	VariantedComputeProgram cp_shift;
	VariantedComputeProgram cp_dft;
	VariantedComputeProgram cp_split;
	VariantedComputeProgram cp_step;

};

#include "FFT_Templated.h"
