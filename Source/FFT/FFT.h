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

	fft_plan create_plan(size_t array_size, size_t supported_max_radix = 200, size_t supported_min_radix = fft_iteration::radix_dft);
	fft_plan create_plan(size_t array_size, const std::vector<size_t>& supported_radixes = { 25, 16, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2 });
	
	template<typename T> void split		(T& source, T& target, glm::ivec3 group_count);
	template<typename T> void step		(T& source, T& target, int32_t radix, fft_dimension dimension = default_fft_dimension<T>());

	bool shaders_are_set = false;

	VariantedComputeProgram cp_copy;
	VariantedComputeProgram cp_shift;
	VariantedComputeProgram cp_dft;
	VariantedComputeProgram cp_split;

};

#include "FFT_Templated.h"

template<typename T>
inline void FFFT2::step(T& source, T& target, int32_t radix, fft_dimension dimension)
{
	compile_shaders();

	if (is_same(source, target)) {
		ASSERT(false);
	}

	if (source.get_size() != target.get_size()) {
		ASSERT(false);
	}

	if (source.get_internal_format_color() != target.get_internal_format_color()) {
		ASSERT(false);
	}

	if (glm::any(glm::lessThan(group_count, glm::ivec3(1)))) {
		ASSERT(false);
	}

	if (glm::any(glm::notEqual(to_ivec3(source.get_size(), 1) % group_count, glm::ivec3(0)))) {
		ASSERT(false);
	}

	cp_split.begin_variant();
	cp_split.variant_define("ffft_source_format", TextureBase2::ColorTextureFormat_to_OpenGL_compute_Image_format(source.get_internal_format_color()));
	cp_split.variant_define("ffft_target_format", TextureBase2::ColorTextureFormat_to_OpenGL_compute_Image_format(target.get_internal_format_color()));
	cp_split.variant_define("source_image", TextureBase2::ColorTextureFormat_to_OpenGL_compute_Image_type<T>(source.get_internal_format_color()));
	cp_split.variant_define("target_image", TextureBase2::ColorTextureFormat_to_OpenGL_compute_Image_type<T>(target.get_internal_format_color()));
	cp_split.variant_define("source_image_dimensionality", std::to_string(TextureBase2::get_texture_dimention<T>()));
	cp_split.variant_define("target_image_dimensionality", std::to_string(TextureBase2::get_texture_dimention<T>()));

	std::string group_count_str = std::string("ivec3(") + std::to_string(group_count.x) + ", " + std::to_string(group_count.y) + ", " + std::to_string(group_count.z) + ")";
	cp_split.variant_define("group_count", group_count_str);

	ComputeProgram& kernel = *cp_split.get_current_variant();

	kernel.update_uniform_as_image("fft_source_texture", source, 0);
	kernel.update_uniform_as_image("fft_target_texture", target, 0);

	kernel.update_uniform("fft_source_texture_resolution", to_ivec3(source.get_size(), 1));
	kernel.update_uniform("fft_target_texture_resolution", to_ivec3(target.get_size(), 1));

	kernel.update_uniform("fft_texture_source_offset", glm::ivec3(0));
	kernel.update_uniform("fft_texture_target_offset", glm::ivec3(0));
	kernel.update_uniform("fft_texture_region", to_ivec3(source.get_size(), 1));

	kernel.dispatch_thread(to_ivec3(source.get_size(), 1));
}


