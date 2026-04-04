#pragma once

#include "Texture1D.h"
#include "Texture2D.h"
#include "Texture2DArray.h"
#include "Texture3D.h"
#include "Buffer.h"

#include "ComputeProgram.h"

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

	template<typename T> void fft		(T&	source,	T& target, fft_dimension dimension = default_fft_dimension<T>(), fft_algorithm algorithm = radix235_fft, glm::ivec3 offset = glm::ivec3(0), glm::ivec3 size = glm::ivec3(0));
	template<typename T> void i_fft		(T&	source,	T& target, fft_dimension dimension = default_fft_dimension<T>(), fft_algorithm algorithm = radix235_fft, glm::ivec3 offset = glm::ivec3(0), glm::ivec3 size = glm::ivec3(0));
	
	template<typename T> void shift		(T&	source,	T& target, glm::ivec3 shift_size, fft_dimension dimension = default_fft_dimension<T>(), glm::ivec3 offset = glm::ivec3(0), glm::ivec3 size = glm::ivec3(0));
	template<typename T> void i_shift	(T& source, T& target, glm::ivec3 shift_size, fft_dimension dimension = default_fft_dimension<T>(), glm::ivec3 offset = glm::ivec3(0), glm::ivec3 size = glm::ivec3(0));

	template<typename T> void copy		(T& source,	T& target, component comp = real_complex, glm::ivec3 offset = glm::ivec3(0), glm::ivec3 size = glm::ivec3(0));
	template<typename T> std::shared_ptr<T> create	(T& source, component comp = real_complex, glm::ivec3 size = glm::ivec3(0));

	template<typename T> std::shared_ptr<T> pad		(T& source, component comp = real_complex, glm::ivec3 padding_size = glm::ivec3(0), glm::vec2 padding_value = glm::vec2(0), glm::ivec3 offset = glm::ivec3(0), glm::ivec3 size = glm::ivec3(0));
	template<typename T> void				i_pad	(T& source, T& target, component comp = real_complex, glm::ivec3 padding_size = glm::ivec3(0), glm::ivec3 offset = glm::ivec3(0), glm::ivec3 size = glm::ivec3(0));
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
	template<typename T> static bool are_same(T& source, T& target);

private:

	enum data_type {
		buffer,
		texture1d,
		texture2d,
		texture2darray,
		texture3d,
	};

	void set_source_type(data_type data_type);
	void set_source_color_texture_format(Texture2D::ColorTextureFormat color_texture_format);

	void set_target_type(data_type data_type);
	void set_target_color_texture_format(Texture2D::ColorTextureFormat color_texture_format);

	template<typename T> std::vector<std::pair<std::string, std::string>> generate_shader_macros();
	template<typename T> void compile_shaders();
	
	Texture2D::ColorTextureFormat complex_texture_format(Texture2D::ColorTextureFormat real_texture);
	Texture2D::ColorTextureFormat real_texture_format(Texture2D::ColorTextureFormat complex_texture);

	template<typename T> void dft		(T& source, T& target, fft_dimension dimension = default_fft_dimension<T>(), glm::ivec3 offset = glm::ivec3(0), glm::ivec3 size = glm::ivec3(0));
	template<typename T> void radix2	(T& source, T& target, fft_dimension dimension = default_fft_dimension<T>(), glm::ivec3 offset = glm::ivec3(0), glm::ivec3 size = glm::ivec3(0));
	template<typename T> void radix3	(T& source, T& target, fft_dimension dimension = default_fft_dimension<T>(), glm::ivec3 offset = glm::ivec3(0), glm::ivec3 size = glm::ivec3(0));
	template<typename T> void radix5	(T& source, T& target, fft_dimension dimension = default_fft_dimension<T>(), glm::ivec3 offset = glm::ivec3(0), glm::ivec3 size = glm::ivec3(0));
	
	bool shaders_are_compiled = false;
	data_type source_type = texture2d;
	data_type target_type = texture2d;
	Texture2D::ColorTextureFormat source_texture_format = Texture2D::ColorTextureFormat::RG16F;
	Texture2D::ColorTextureFormat target_texture_format = Texture2D::ColorTextureFormat::RG16F;

	std::shared_ptr<ComputeProgram> cp_dft_x;
	std::shared_ptr<ComputeProgram> cp_dft_y;
	std::shared_ptr<ComputeProgram> cp_dft_z;
};

#include "FFT_Templated.h"

template<typename T>
inline bool FFFT2::are_same(T& source, T& target)
{
	return source.id == target.id;
}

namespace {
	template<typename texture_type, typename vector_type>
	std::shared_ptr<texture_type> create_texture_glm(vector_type size, Texture2D::ColorTextureFormat format) = delete;

	template<> std::shared_ptr<Texture1D> create_texture_glm(glm::ivec1 size, Texture2D::ColorTextureFormat format) {
		return std::make_shared<Texture1D>(size.x, format, 1, 0);
	}
	template<> std::shared_ptr<Texture2D> create_texture_glm(glm::ivec2 size, Texture2D::ColorTextureFormat format) {
		return std::make_shared<Texture2D>(size.x, size.y, format, 1, 0);
	}
	template<> std::shared_ptr<Texture2DArray> create_texture_glm(glm::ivec3 size, Texture2D::ColorTextureFormat format) {
		return std::make_shared<Texture2DArray>(size.x, size.y, size.z, format, 1, 0);
	}
	template<> std::shared_ptr<Texture3D> create_texture_glm(glm::ivec3 size, Texture2D::ColorTextureFormat format) {
		return std::make_shared<Texture3D>(size.x, size.y, size.z, format, 1, 0);
	}

	glm::ivec3 to_ivec3(glm::ivec2 size2) {
		return glm::ivec3(size2.x, size2.y, 0);
	}

	glm::ivec3 to_ivec3(glm::ivec1 size1) {
		return glm::ivec3(size1.x, 0, 0);
	}
}

template<typename T>
inline void FFFT2::copy(T& source, T& target, component comp, glm::ivec3 offset, glm::ivec3 size)
{
	
}

template<typename T>
inline std::shared_ptr<T> FFFT2::create(T& source, component comp, glm::ivec3 size)
{
	Texture2D::ColorTextureFormat format = source.get_internal_format_color();
	if (comp == real_complex)	format = complex_texture_format(format);
	if (comp == real)			format = real_texture_format(format);
	if (comp == complex)		format = complex_texture_format(format);

	if (size.x <= 0) size.x = to_ivec3(source.get_size()).x;
	if (size.y <= 0) size.y = to_ivec3(source.get_size()).y;
	if (size.z <= 0) size.z = to_ivec3(source.get_size()).z;

	return create_texture_glm<T>(source.get_size(), format);
}