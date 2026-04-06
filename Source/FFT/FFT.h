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
	
	template<typename T> void				shift	(T&	source,	T& target, glm::ivec3 shift_size, fft_dimension dimension = default_fft_dimension<T>(), glm::ivec3 offset = glm::ivec3(0), glm::ivec3 size = glm::ivec3(0));
	template<typename T> void				i_shift	(T& source, T& target, glm::ivec3 shift_size, fft_dimension dimension = default_fft_dimension<T>(), glm::ivec3 offset = glm::ivec3(0), glm::ivec3 size = glm::ivec3(0));

	template<typename T> void				copy	(T& source,	T& target, component comp = real_complex, glm::ivec3 source_offset = glm::ivec3(0), glm::ivec3 target_offset = glm::ivec3(0), glm::ivec3 size = glm::ivec3(0));
	template<typename T> std::shared_ptr<T> create	(T& source, component comp = real_complex, glm::ivec3 size = glm::ivec3(0));

	template<typename T> void				pad		(T& source, T& target, glm::vec2 padding_value = glm::vec2(0), glm::ivec3 offset = glm::ivec3(0));
	template<typename T> void				i_pad	(T& source, T& target, glm::ivec3 offset = glm::ivec3(0));
	template<typename T> std::shared_ptr<T> pad		(T& source, component comp = real_complex, glm::ivec3 padding_size = glm::ivec3(0), glm::vec2 padding_value = glm::vec2(0), glm::ivec3 offset = glm::ivec3(0), glm::ivec3 size = glm::ivec3(0));
	template<typename T> std::shared_ptr<T> i_pad	(T& source, component comp = real_complex, glm::vec2 padding_value = glm::vec2(0), glm::ivec3 padding_size = glm::ivec3(0), glm::ivec3 offset = glm::ivec3(0), glm::ivec3 size = glm::ivec3(0));

	//void window();
	//void inverse_window();
	//void conjugate();
	//void multiply();
	//void divide();
	//void add();
	//void subtract();
	
	template<typename T> static bool is_complex(T& source);
	template<typename T> static bool is_real(T& source);
	template<typename T> static bool is_same(T& source, T& target);

private:

	std::string component_to_string(component component);

	Texture2D::ColorTextureFormat complex_texture_format(Texture2D::ColorTextureFormat real_texture);
	Texture2D::ColorTextureFormat real_texture_format(Texture2D::ColorTextureFormat complex_texture);

	template<typename T> void dft		(T& source, T& target, fft_dimension dimension = default_fft_dimension<T>(), glm::ivec3 offset = glm::ivec3(0), glm::ivec3 size = glm::ivec3(0));
	template<typename T> void radix2	(T& source, T& target, fft_dimension dimension = default_fft_dimension<T>(), glm::ivec3 offset = glm::ivec3(0), glm::ivec3 size = glm::ivec3(0));
	template<typename T> void radix3	(T& source, T& target, fft_dimension dimension = default_fft_dimension<T>(), glm::ivec3 offset = glm::ivec3(0), glm::ivec3 size = glm::ivec3(0));
	template<typename T> void radix5	(T& source, T& target, fft_dimension dimension = default_fft_dimension<T>(), glm::ivec3 offset = glm::ivec3(0), glm::ivec3 size = glm::ivec3(0));
	
	VariantedComputeProgram cp_dft;
	VariantedComputeProgram cp_copy;
};

#include "FFT_Templated.h"

template<typename texture_type, typename vector_type>
void clear_texture_glm(texture_type& texture, vector_type offset, vector_type size, glm::vec4 color) = delete;

template<> void clear_texture_glm(Texture1D& texture, glm::ivec3 offset, glm::ivec3 size, glm::vec4 color) {
	texture.clear(color, offset.x, size.x, 0);
}
template<> void clear_texture_glm(Texture2D& texture, glm::ivec3 offset, glm::ivec3 size, glm::vec4 color) {
	texture.clear(color, offset.x, offset.y, size.x, size.y, 0);
}
template<> void clear_texture_glm(Texture2DArray& texture, glm::ivec3 offset, glm::ivec3 size, glm::vec4 color) {
	texture.clear(color, offset.x, offset.y, offset.z, size.x, size.y, size.z, 0);
}
template<> void clear_texture_glm(Texture3D& texture, glm::ivec3 offset, glm::ivec3 size, glm::vec4 color) {
	texture.clear(color, offset.x, offset.y, offset.z, size.x, size.y, size.z, 0);
}

template<typename T>
inline void FFFT2::pad(T& source, T& target, glm::vec2 padding_value, glm::ivec3 offset)
{
	if (glm::any(glm::lessThan(target.get_size(), source.get_size()))) {
		ASSERT(false);
	}

	if (glm::any(glm::greaterThan(to_ivec3(source.get_size()) + offset, to_ivec3(target.get_size())))) {
		ASSERT(false);
	}


	clear_texture_glm(target, glm::ivec3(0), to_ivec3(target.get_size()), glm::vec4(padding_value, 0, 1));
	copy()

}

