#pragma once

#include "Texture1D.h"
#include "Texture2D.h"
#include "Texture2DArray.h"
#include "Texture3D.h"
#include "Buffer.h"

#include "ComputeProgram.h"
#include "Tools/VariantedComputeProgram/VariantedComputeProgram.h"

#include <filesystem>

namespace shader_directory {
	static std::filesystem::path ffft2_shader_directory = "../FFFT/Source/GLSL/FFT/";
}

// Fast & Furious Fourier Transform
class FFFT2 {
public:
	
	enum fft_dimension {
		x	= 0b001,
		y	= 0b010,
		z	= 0b100,
		xy	= x | y,
		yz	= y | z,
		xz	= x | z,
		xyz = x | y | z,
	};

	template<typename T>
	constexpr static fft_dimension default_fft_dimension();

	enum windowing_function {
		hann,
		hamming,
	};

	enum component {
		real,
		complex,
		real_complex,
	};

	template<typename T> void				fft		(T&	source,	T& target, fft_dimension dimension = default_fft_dimension<T>(), size_t max_radix = 32);
	template<typename T> void				i_fft	(T&	source,	T& target, fft_dimension dimension = default_fft_dimension<T>(), size_t max_radix = 32);
	
	template<typename T> void				shift	(T&	source,	T& target, glm::ivec3 shift_size);
	template<typename T> std::shared_ptr<T>	shift	(T&	source, glm::ivec3 shift_size);
	template<typename T> void				shift	(T& source, T& target, fft_dimension dimension);
	template<typename T> std::shared_ptr<T>	shift	(T& source, fft_dimension dimension);
	template<typename T> void				i_shift	(T& source, T& target, glm::ivec3 shift_size);
	template<typename T> std::shared_ptr<T>	i_shift	(T& source, glm::ivec3 shift_size);
	template<typename T> void				i_shift	(T& source, T& target, fft_dimension dimension);
	template<typename T> std::shared_ptr<T>	i_shift	(T& source, fft_dimension dimension);

	template<typename T> void				copy	(T& source,	T& target, component comp = real_complex, glm::ivec3 source_offset = glm::ivec3(0), glm::ivec3 target_offset = glm::ivec3(0), glm::ivec3 size = glm::ivec3(0));
	template<typename T> std::shared_ptr<T> create	(T& source, component comp = real_complex, glm::ivec3 size = glm::ivec3(0));
	template<typename T> std::shared_ptr<T> create	(TextureBase2::ColorTextureFormat format, component comp = real_complex, glm::ivec3 size = glm::ivec3(0));

	template<typename T> void				pad		(T& source, T& target, glm::ivec3 offset = glm::ivec3(0), glm::vec2 padding_value = glm::vec2(0));
	template<typename T> void				i_pad	(T& source, T& target, glm::ivec3 offset = glm::ivec3(0));
	template<typename T> std::shared_ptr<T> pad		(T& source, glm::ivec3 padded_size, glm::ivec3 offset = glm::ivec3(0), glm::vec2 padding_value = glm::vec2(0));
	template<typename T> std::shared_ptr<T> i_pad	(T& source, glm::ivec3 offset, glm::ivec3 size);

	template<typename T> void				op			(T& source, glm::vec4 constant, std::string operation, std::string additional_definition = "");
	template<typename T> void				conjugate	(T& source);
	template<typename T> void				multiply	(T& source, glm::vec2 constant);
	template<typename T> void				divide		(T& source, glm::vec2 constant);
	template<typename T> void				add			(T& source, glm::vec2 constant);
	template<typename T> void				subtract	(T& source, glm::vec2 constant);

	template<typename T> void				taper_tukey	(T& source, fft_dimension dimension, float alpha = 0.01f);
	//void window();
	//void inverse_window();


private:

	template<typename T> static bool is_complex(T& source);
	template<typename T> static bool is_real(T& source);
	template<typename T> static bool is_same(T& source, T& target);

	std::string component_to_string(component component);

	Texture2D::ColorTextureFormat complex_texture_format(Texture2D::ColorTextureFormat real_texture);
	Texture2D::ColorTextureFormat real_texture_format(Texture2D::ColorTextureFormat complex_texture);

	void compile_shaders();

	struct fft_iteration {
		constexpr static size_t radix_dft = 1;
		size_t	radix		= radix_dft;
		size_t	chunk_size	= 0;
	};

	struct fft_plan {
		std::vector<fft_iteration> iterations;
	};

	fft_plan create_plan(size_t array_size, size_t supported_max_radix, size_t supported_min_radix = fft_iteration::radix_dft);
	fft_plan create_plan(size_t array_size, const std::vector<size_t>& supported_radixes = { 25, 16, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2 });
	
	template<typename T> void mixed_fft	(T& source, T& target, fft_dimension dimension = default_fft_dimension<T>(), size_t max_radix = 32, bool inverse = false);
	template<typename T> void split		(T& source, T& target, glm::ivec3 split_count, glm::ivec3 group_count = glm::ivec3(1));
	template<typename T> void step		(T& source, T& target, size_t radix, fft_dimension dimension = default_fft_dimension<T>(), bool inverse = false, glm::ivec3 group_count = glm::ivec3(1));
	template<typename T> void dft		(T& source, T& target, fft_dimension dimension = default_fft_dimension<T>(), bool inverse = false, glm::ivec3 group_count = glm::ivec3(1));

	bool shaders_are_set = false;

	VariantedComputeProgram cp_copy;
	VariantedComputeProgram cp_shift;
	VariantedComputeProgram cp_dft;
	VariantedComputeProgram cp_split;
	VariantedComputeProgram cp_step;
	VariantedComputeProgram cp_op;
	VariantedComputeProgram cp_window;

	glm::ivec3 to_ivec3(glm::ivec3 size3, int32_t blank_value = 0);
	glm::ivec3 to_ivec3(glm::ivec2 size2, int32_t blank_value = 0);
	glm::ivec3 to_ivec3(glm::ivec1 size1, int32_t blank_value = 0);
	template<typename texture_type>
	std::shared_ptr<texture_type> create_texture_glm(glm::ivec3 size, Texture2D::ColorTextureFormat format) = delete;
	template<typename texture_type, typename vector_type>
	void clear_texture_glm(texture_type& texture, vector_type offset, vector_type size, glm::vec4 color) = delete;
};

#include "FFT_Templated.h"
