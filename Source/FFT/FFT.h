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
	//extern std::filesystem::path ffft2_shader_directory;
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

	template<typename T> void				fft		(T&	source,	T& target, fft_dimension dimension = default_fft_dimension<T>(), fft_algorithm algorithm = radix235_fft, glm::ivec3 offset = glm::ivec3(0), glm::ivec3 size = glm::ivec3(0));
	template<typename T> void				i_fft	(T&	source,	T& target, fft_dimension dimension = default_fft_dimension<T>(), fft_algorithm algorithm = radix235_fft, glm::ivec3 offset = glm::ivec3(0), glm::ivec3 size = glm::ivec3(0));
	
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

private:

	template<typename T> static bool is_complex(T& source);
	template<typename T> static bool is_real(T& source);
	template<typename T> static bool is_same(T& source, T& target);

	std::string component_to_string(component component);

	Texture2D::ColorTextureFormat complex_texture_format(Texture2D::ColorTextureFormat real_texture);
	Texture2D::ColorTextureFormat real_texture_format(Texture2D::ColorTextureFormat complex_texture);

	void compile_shaders();

	template<typename T> void dft		(T& source, T& target, fft_dimension dimension = default_fft_dimension<T>(), glm::ivec3 offset = glm::ivec3(0), glm::ivec3 size = glm::ivec3(0));
	template<typename T> void radix2	(T& source, T& target, fft_dimension dimension = default_fft_dimension<T>(), glm::ivec3 offset = glm::ivec3(0), glm::ivec3 size = glm::ivec3(0));
	template<typename T> void radix3	(T& source, T& target, fft_dimension dimension = default_fft_dimension<T>(), glm::ivec3 offset = glm::ivec3(0), glm::ivec3 size = glm::ivec3(0));
	template<typename T> void radix4	(T& source, T& target, fft_dimension dimension = default_fft_dimension<T>(), glm::ivec3 offset = glm::ivec3(0), glm::ivec3 size = glm::ivec3(0));
	template<typename T> void radix5	(T& source, T& target, fft_dimension dimension = default_fft_dimension<T>(), glm::ivec3 offset = glm::ivec3(0), glm::ivec3 size = glm::ivec3(0));
	template<typename T> void radix6	(T& source, T& target, fft_dimension dimension = default_fft_dimension<T>(), glm::ivec3 offset = glm::ivec3(0), glm::ivec3 size = glm::ivec3(0));
	template<typename T> void radix7	(T& source, T& target, fft_dimension dimension = default_fft_dimension<T>(), glm::ivec3 offset = glm::ivec3(0), glm::ivec3 size = glm::ivec3(0));
	template<typename T> void radix8	(T& source, T& target, fft_dimension dimension = default_fft_dimension<T>(), glm::ivec3 offset = glm::ivec3(0), glm::ivec3 size = glm::ivec3(0));
	template<typename T> void radix9	(T& source, T& target, fft_dimension dimension = default_fft_dimension<T>(), glm::ivec3 offset = glm::ivec3(0), glm::ivec3 size = glm::ivec3(0));
	template<typename T> void radix10	(T& source, T& target, fft_dimension dimension = default_fft_dimension<T>(), glm::ivec3 offset = glm::ivec3(0), glm::ivec3 size = glm::ivec3(0));
	template<typename T> void radix11	(T& source, T& target, fft_dimension dimension = default_fft_dimension<T>(), glm::ivec3 offset = glm::ivec3(0), glm::ivec3 size = glm::ivec3(0));
	template<typename T> void radix16	(T& source, T& target, fft_dimension dimension = default_fft_dimension<T>(), glm::ivec3 offset = glm::ivec3(0), glm::ivec3 size = glm::ivec3(0));
	template<typename T> void radix25	(T& source, T& target, fft_dimension dimension = default_fft_dimension<T>(), glm::ivec3 offset = glm::ivec3(0), glm::ivec3 size = glm::ivec3(0));
	
	bool shaders_are_set = false;

	VariantedComputeProgram cp_copy;
	VariantedComputeProgram cp_shift;
	VariantedComputeProgram cp_dft;
};

#include "FFT_Templated.h"

template<typename T>
inline void FFFT2::copy(T& source, T& target, component comp, glm::ivec3 source_offset, glm::ivec3 target_offset, glm::ivec3 size)
{
	compile_shaders();

	if (is_same(source, target)) {
		std::cout << "[FFFT Error] FFFT::copy() is called with identical source and target but self-copy is not supported" << std::endl;
		ASSERT(false);
		return;
	}

	if (!is_complex(source) && !is_real(source)) {
		std::cout << "[FFFT Error] FFFT::copy() is called with a source that is neither of real or complex type" << std::endl;
		ASSERT(false);
	}

	if (!is_complex(target) && !is_real(target)) {
		std::cout << "[FFFT Error] FFFT::copy() is called with a target that is neither of real or complex type" << std::endl;
		ASSERT(false);
	}

	if (is_real(source) && is_real(target) && comp == real)
		comp = real_complex;

	if (is_real(source) && is_real(target) && comp == complex)
		return;

	if (size.x == 0) size.x = to_ivec3(glm::min(source.get_size(), target.get_size()), 1).x - max(source_offset, target_offset).x;
	if (size.y == 0) size.y = to_ivec3(glm::min(source.get_size(), target.get_size()), 1).y - max(source_offset, target_offset).y;
	if (size.z == 0) size.z = to_ivec3(glm::min(source.get_size(), target.get_size()), 1).z - max(source_offset, target_offset).z;

	bool source_overflow = glm::any(glm::greaterThan(source_offset + size, to_ivec3(source.get_size(), 1)));
	bool target_overflow = glm::any(glm::greaterThan(target_offset + size, to_ivec3(target.get_size(), 1)));

	if (source_overflow || target_overflow) {
		std::cout << "[FFFT Error] FFFT::copy() is called but specified offset and size exceeds data size" << std::endl;
		ASSERT(false);
		return;
	}

	cp_copy.begin_variant();
	cp_copy.variant_define("ffft_source_format",	TextureBase2::ColorTextureFormat_to_OpenGL_compute_Image_format(source.get_internal_format_color()));
	cp_copy.variant_define("ffft_target_format",	TextureBase2::ColorTextureFormat_to_OpenGL_compute_Image_format(target.get_internal_format_color()));
	cp_copy.variant_define("source_image",			TextureBase2::ColorTextureFormat_to_OpenGL_compute_Image_type<T>(source.get_internal_format_color()));
	cp_copy.variant_define("target_image",			TextureBase2::ColorTextureFormat_to_OpenGL_compute_Image_type<T>(target.get_internal_format_color()));
	cp_copy.variant_define("source_image_dimensionality", std::to_string(TextureBase2::get_texture_dimention<T>()));
	cp_copy.variant_define("target_image_dimensionality", std::to_string(TextureBase2::get_texture_dimention<T>()));

	cp_copy.variant_define("copy_operation", component_to_string(comp));

	ComputeProgram& kernel = *cp_copy.get_current_variant();

	kernel.update_uniform_as_image("fft_source_texture", source, 0);
	kernel.update_uniform_as_image("fft_target_texture", target, 0);

	kernel.update_uniform("fft_source_texture_resolution", to_ivec3(source.get_size(), 1));
	kernel.update_uniform("fft_target_texture_resolution", to_ivec3(target.get_size(), 1));

	kernel.update_uniform("fft_texture_source_offset", source_offset);
	kernel.update_uniform("fft_texture_target_offset", target_offset);
	kernel.update_uniform("fft_texture_region", size);

	kernel.dispatch_thread(size);
}


template<typename T>
inline bool FFFT2::is_complex(T& source)
{
	return source.ColorTextureFormat_channels(source.get_internal_format_color()) == 2 && (source.ColorTextureFormat_to_Type(source.get_internal_format_color()) == Texture3D::Type::HALF_FLOAT || source.ColorTextureFormat_to_Type(source.get_internal_format_color()) == Texture3D::Type::FLOAT);;
}

template<typename T>
inline bool FFFT2::is_real(T& source)
{
	return source.ColorTextureFormat_channels(source.get_internal_format_color()) == 1 && (source.ColorTextureFormat_to_Type(source.get_internal_format_color()) == Texture3D::Type::HALF_FLOAT || source.ColorTextureFormat_to_Type(source.get_internal_format_color()) == Texture3D::Type::FLOAT);;
}

