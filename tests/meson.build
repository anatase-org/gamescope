catch2_dep = dependency('catch2-with-main', required: false)

srcdir = '../src'

test_frame_generation_config = executable(
	'test_frame_generation_config',
	[srcdir / 'frame_generation_config.cpp', 'test_frame_generation_config.cpp'],
	include_directories: [srcdir],
	cpp_args: gamescope_cpp_args,
)
test('frame-generation-config', test_frame_generation_config)

test('benchmark-tools', find_program('python3', native: true),
  args: [files('test_benchmark.py')])

test_surface_damage = executable(
	'test_surface_damage',
	'test_surface_damage.cpp',
	include_directories: [srcdir],
	dependencies: [wlroots_dep],
	cpp_args: gamescope_cpp_args,
)
test('surface-damage', test_surface_damage)

test_vblank_schedule = executable(
	'test_vblank_schedule',
	'test_vblank_schedule.cpp',
	include_directories: [srcdir],
	cpp_args: gamescope_cpp_args,
)
test('vblank-schedule', test_vblank_schedule)

if catch2_dep.found()
	unittest_src = []
	foreach src : gamescope_core_src + ['Script/Script.cpp', 'convar_script.cpp']
		unittest_src += srcdir / src
	endforeach

	test_convar = executable(
		'test_convar',
		unittest_src + ['test_convar.cpp'],
		include_directories: [srcdir, sol2_include],
		dependencies: [catch2_dep, luajit_dep, cap_dep, drm_dep, glm_dep, wlroots_dep],
		cpp_args: gamescope_cpp_args,
	)
	test('convar', test_convar)
	reaper_test_helper = executable('reaper_test_helper', 'reaper_test_helper.cpp')
	test_process = executable(
		'test_process', unittest_src + ['test_process.cpp'],
		include_directories: [srcdir, sol2_include],
		dependencies: [catch2_dep, luajit_dep, cap_dep, drm_dep, glm_dep, wlroots_dep],
		cpp_args: gamescope_cpp_args,
	)
	test('utils/process', test_process, depends: [reaper_test_helper])
	test_upscale = executable('test_upscale', 'test_upscale.cpp',
		include_directories: [srcdir], dependencies: [catch2_dep], cpp_args: gamescope_cpp_args)
	test('upscale', test_upscale)

	foreach name : ['mangoapp_config', 'mangoapp_control', 'mode_list']
		t = executable('test_' + name, 'test_' + name + '.cpp',
			include_directories: [srcdir], dependencies: [catch2_dep], cpp_args: gamescope_cpp_args)
		test(name, t)
	endforeach

endif

# Keep the color regression available when Catch2 is not installed.
test_color_helpers = executable(
	'test_color_helpers',
	['test_color_helpers.cpp', srcdir / 'color_helpers.cpp'],
	include_directories: [srcdir],
	dependencies: [glm_dep],
	cpp_args: gamescope_cpp_args,
)
test('color-helpers', test_color_helpers)
