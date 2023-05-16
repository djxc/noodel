use std::sync::Arc;

use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer, BufferContents},
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo, QueueFlags, Queue,
    },
    image::{ImageUsage, SwapchainImage, view::ImageView},
    instance::{Instance, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryUsage, StandardMemoryAllocator},
    swapchain::{AcquireError, Surface, Swapchain, SwapchainCreateInfo, SwapchainCreationError, self, SwapchainPresentInfo},
    sync::{future::FenceSignalFuture, FlushError, self, GpuFuture},
    VulkanLibrary, pipeline::{graphics::{viewport::{Viewport, ViewportState}, vertex_input::Vertex, input_assembly::InputAssemblyState}, GraphicsPipeline}, render_pass::{RenderPass, Framebuffer, FramebufferCreateInfo, Subpass}, command_buffer::{PrimaryAutoCommandBuffer, allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo}, AutoCommandBufferBuilder, CommandBufferUsage, RenderPassBeginInfo, SubpassContents}, shader::ShaderModule,
};
use vulkano_win::VkSurfaceBuild;
use winit::{
    event::{Event, VirtualKeyCode, WindowEvent, KeyboardInput, ElementState},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

use crate::{conf::index_conf, DGIS};


#[derive(BufferContents, Vertex)]
#[repr(C)]
struct DVertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
}


mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
            #version 460

            layout(location = 0) in vec2 position;

            void main() {
                gl_Position = vec4(position, 0.0, 1.0);
            }
        ",
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
            #version 460

            layout(location = 0) out vec4 f_color;

            void main() {
                f_color = vec4(1.0, 0.0, 0.0, 1.0);
            }
        ",
    }
}

mod cs {
    vulkano_shaders::shader!{
        ty: "compute",
        src: r"
            #version 460

            layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

            layout(set = 0, binding = 0) buffer Data {
                uint data[];
            } buf;

            void main() {
                uint idx = gl_GlobalInvocationID.x;
                buf.data[idx] *= 12;
            }
        ",
    }
}


/// 通过win窗口显示vulkan渲染结果
///
/// 1、首先创建带有win扩展的实例
/// 2、创建窗口，以及事件循环
struct DMapRender {}

/// 创建带有win扩展的实例
fn create_win_instance() -> Arc<Instance> {
    let library = VulkanLibrary::new().expect("no local Vulkan library/DLL");
    let required_extensions = vulkano_win::required_extensions(&library);
    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            enabled_extensions: required_extensions,
            ..Default::default()
        },
    )
    .expect("failed to create instance");
    return instance;
}

/// 选择物理设备
fn select_physical_device(
    instance: &Arc<Instance>,
    surface: &Arc<Surface>,
    device_extensions: &DeviceExtensions,
) -> (Arc<PhysicalDevice>, u32) {
    instance
        .enumerate_physical_devices()
        .expect("could not enumerate devices")
        .filter(|p| p.supported_extensions().contains(&device_extensions))
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .enumerate()
                // Find the first first queue family that is suitable.
                // If none is found, `None` is returned to `filter_map`,
                // which disqualifies this physical device.
                .position(|(i, q)| {
                    q.queue_flags.contains(QueueFlags::GRAPHICS)
                        && p.surface_support(i as u32, &surface).unwrap_or(false)
                })
                .map(|q| (p, q as u32))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,

            // Note that there exists `PhysicalDeviceType::Other`, however,
            // `PhysicalDeviceType` is a non-exhaustive enum. Thus, one should
            // match wildcard `_` to catch all unknown device types.
            _ => 4,
        })
        .expect("no device available")
}

fn get_render_pass(device: Arc<Device>, swapchain: &Arc<Swapchain>) -> Arc<RenderPass> {
    vulkano::single_pass_renderpass!(
        device,
        attachments: {
            color: {
                load: Clear,
                store: Store,
                format: swapchain.image_format(), // set the format the same as the swapchain
                samples: 1,
            },
        },
        pass: {
            color: [color],
            depth_stencil: {},
        },
    )
    .unwrap()
}


fn get_pipeline(
    device: Arc<Device>,
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
    render_pass: Arc<RenderPass>,
    viewport: Viewport,
) -> Arc<GraphicsPipeline> {
    GraphicsPipeline::start()
        .vertex_input_state(DVertex::per_vertex())
        .vertex_shader(vs.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new())
        .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([viewport]))
        .fragment_shader(fs.entry_point("main").unwrap(), ())
        .render_pass(Subpass::from(render_pass, 0).unwrap())
        .build(device)
        .unwrap()
}


fn get_command_buffers(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    pipeline: &Arc<GraphicsPipeline>,
    framebuffers: &Vec<Arc<Framebuffer>>,
    vertex_buffer: &Subbuffer<[DVertex]>,
) -> Vec<Arc<PrimaryAutoCommandBuffer>> {

    framebuffers
        .iter()
        .map(|framebuffer| {
            let command_buffer_allocator = StandardCommandBufferAllocator::new(
                device.clone(),
                StandardCommandBufferAllocatorCreateInfo::default(),
            );
            let mut builder = AutoCommandBufferBuilder::primary(
                &command_buffer_allocator,
                queue.queue_family_index(),
                CommandBufferUsage::MultipleSubmit, // don't forget to write the correct buffer usage
            )
            .unwrap();

            builder
                .begin_render_pass(
                    RenderPassBeginInfo {
                        clear_values: vec![Some([0.1, 0.1, 0.1, 1.0].into())],
                        ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                    },
                    SubpassContents::Inline,
                )
                .unwrap()
                .bind_pipeline_graphics(pipeline.clone())
                .bind_vertex_buffers(0, vertex_buffer.clone())
                .draw(vertex_buffer.len() as u32, 1, 0, 0)
                .unwrap()
                .end_render_pass()
                .unwrap();

            Arc::new(builder.build().unwrap())
        })
        .collect()
}


/// 创建帧缓存
fn get_framebuffers(
    images: &[Arc<SwapchainImage>],
    render_pass: &Arc<RenderPass>,
) -> Vec<Arc<Framebuffer>> {
    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
}

impl DMapRender {

    fn init(self) {
        let instance = create_win_instance();

        let event_loop = EventLoop::new();

        let window_builder = WindowBuilder::new();
        let surface = window_builder
            .build_vk_surface(&event_loop, instance.clone())
            .unwrap();

        let window = surface
            .object()
            .unwrap()
            .clone()
            .downcast::<Window>()
            .unwrap();
        window.set_title(index_conf::CONF_MAP.get("winit_name").unwrap());

        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };

        let (physical_device, queue_family_index) =
            select_physical_device(&instance, &surface, &device_extensions);

        let (device, mut queues) = Device::new(
            physical_device.clone(),
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                enabled_extensions: device_extensions,
                ..Default::default()
            },
        )
        .expect("failed to create device");

        let queue = queues.next().unwrap();

        let caps = physical_device
            .surface_capabilities(&surface, Default::default())
            .expect("failed to get surface capabilities");
        let dimensions = window.inner_size();
        let composite_alpha = caps.supported_composite_alpha.into_iter().next().unwrap();
        let image_format = Some(
            physical_device
                .surface_formats(&surface, Default::default())
                .unwrap()[0]
                .0,
        );

        let (mut swapchain, images) = Swapchain::new(
            device.clone(),
            surface.clone(),
            SwapchainCreateInfo {
                min_image_count: caps.min_image_count + 1, // How many buffers to use in the swapchain
                image_format,
                image_extent: dimensions.into(),
                image_usage: ImageUsage::COLOR_ATTACHMENT, // What the images are going to be used for
                composite_alpha,
                ..Default::default()
            },
        )
        .unwrap();
        let render_pass = get_render_pass(device.clone(), &swapchain);

        let framebuffers = get_framebuffers(&images, &render_pass);

        let vertex1 = DVertex {
            position: [-0.5, -0.5],
        };
        let vertex2 = DVertex {
            position: [0.0, 0.5],
        };
        let vertex3 = DVertex {
            position: [0.5, -0.25],
        };

        let memory_allocator = StandardMemoryAllocator::new_default(device.clone());

        let vertex_buffer = Buffer::from_iter(
            &memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::Upload,
                ..Default::default()
            },
            vec![vertex1, vertex2, vertex3],
        )
        .unwrap();

        let vs = vs::load(device.clone()).expect("failed to create shader module");
        let fs = fs::load(device.clone()).expect("failed to create shader module");
        let mut viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions: window.inner_size().into(),
            depth_range: 0.0..1.0,
        };

        let pipeline = get_pipeline(
            device.clone(),
            vs.clone(),
            fs.clone(),
            render_pass.clone(),
            viewport.clone(),
        );
        let mut command_buffers =
            get_command_buffers(&device, &queue, &pipeline, &framebuffers, &vertex_buffer);

        let mut window_resized = false;
        let mut recreate_swapchain = false;

        let frames_in_flight = images.len();
        let mut fences: Vec<Option<Arc<FenceSignalFuture<_>>>> = vec![None; frames_in_flight];
        let mut previous_fence_i = 0;
        let mut cursor_idx = 0;

        event_loop.run(move |event, _, control_flow| {
            match event {
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    ..
                } => {
                    *control_flow = ControlFlow::Exit;
                }
                // 键盘按下事件
                Event::WindowEvent {
                    event:
                        WindowEvent::KeyboardInput {
                            input:
                                KeyboardInput {
                                    state: ElementState::Pressed,
                                    virtual_keycode: _virtual_key_code,
                                    ..
                                },
                            ..
                        },
                    ..
                } => {
                    let virtual_keycode = _virtual_key_code.unwrap();
                    match virtual_keycode {
                        // A 修改鼠标样式
                        VirtualKeyCode::A => {
                            println!("{:?}", _virtual_key_code.unwrap());
                            // cursor_idx = change_window_cursors(cursor_idx);
                            // window.set_cursor_icon(CURSORS[cursor_idx]);
                        }
                        // B 修改绘制几何
                        VirtualKeyCode::B => {
                            // 1、首先创建swapchain交换链
                            // 2、创建帧缓存
                            // 3、几何数据准备
                            // 4、根据几何数据创建顶点缓冲区
                            // 5、创建图形管道
                            // 6、创建命令缓冲区
                            // 最后图形的绘制在winitclear事件中，异步执行命令
                            let (new_swapchain, new_images) = swapchain
                                .recreate(SwapchainCreateInfo {
                                    image_extent: window.inner_size().into(),
                                    ..swapchain.create_info()
                                })
                                .unwrap();
                            swapchain = new_swapchain;
                            let new_framebuffers =
                                get_framebuffers(&new_images, &render_pass.clone());

                            let vertex1 = DVertex {
                                position: [0.5, 0.5],
                            };
                            let vertex2 = DVertex {
                                position: [0.0, 0.5],
                            };
                            let vertex3 = DVertex {
                                position: [0.5, -0.25],
                            };

                            let vertex_buffer = Buffer::from_iter(
                                &memory_allocator,
                                BufferCreateInfo {
                                    usage: BufferUsage::VERTEX_BUFFER,
                                    ..Default::default()
                                },
                                AllocationCreateInfo {
                                    usage: MemoryUsage::Upload,
                                    ..Default::default()
                                },
                                vec![vertex1, vertex2, vertex3],
                            )
                            .unwrap();
                            let new_pipeline = get_pipeline(
                                device.clone(),
                                vs.clone(),
                                fs.clone(),
                                render_pass.clone(),
                                viewport.clone(),
                            );
                            command_buffers = get_command_buffers(
                                &device,
                                &queue,
                                &new_pipeline,
                                &new_framebuffers,
                                &vertex_buffer,
                            );
                        }
                        _ => {}
                    }
                }
                Event::MainEventsCleared => {
                    if window_resized || recreate_swapchain {
                        recreate_swapchain = false;

                        let new_dimensions = window.inner_size();

                        let (new_swapchain, new_images) =
                            match swapchain.recreate(SwapchainCreateInfo {
                                image_extent: new_dimensions.into(),
                                ..swapchain.create_info()
                            }) {
                                Ok(r) => r,
                                Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => {
                                    return
                                }
                                Err(e) => panic!("failed to recreate swapchain: {e}"),
                            };
                        swapchain = new_swapchain;
                        let new_framebuffers = get_framebuffers(&new_images, &render_pass.clone());

                        if window_resized {
                            window_resized = false;

                            viewport.dimensions = new_dimensions.into();
                            let new_pipeline = get_pipeline(
                                device.clone(),
                                vs.clone(),
                                fs.clone(),
                                render_pass.clone(),
                                viewport.clone(),
                            );
                            command_buffers = get_command_buffers(
                                &device,
                                &queue,
                                &new_pipeline,
                                &new_framebuffers,
                                &vertex_buffer,
                            );
                        }
                    }

                    let (image_i, suboptimal, acquire_future) =
                        match swapchain::acquire_next_image(swapchain.clone(), None) {
                            Ok(r) => r,
                            Err(AcquireError::OutOfDate) => {
                                recreate_swapchain = true;
                                return;
                            }
                            Err(e) => panic!("failed to acquire next image: {e}"),
                        };

                    if suboptimal {
                        recreate_swapchain = true;
                    }

                    // wait for the fence related to this image to finish (normally this would be the oldest fence)
                    if let Some(image_fence) = &fences[image_i as usize] {
                        image_fence.wait(None).unwrap();
                    }

                    let previous_future = match fences[previous_fence_i as usize].clone() {
                        // Create a NowFuture
                        None => {
                            let mut now = sync::now(device.clone());
                            now.cleanup_finished();

                            now.boxed()
                        }
                        // Use the existing FenceSignalFuture
                        Some(fence) => fence.boxed(),
                    };

                    let future = previous_future
                        .join(acquire_future)
                        .then_execute(queue.clone(), command_buffers[image_i as usize].clone())
                        .unwrap()
                        .then_swapchain_present(
                            queue.clone(),
                            SwapchainPresentInfo::swapchain_image_index(swapchain.clone(), image_i),
                        )
                        .then_signal_fence_and_flush();

                    fences[image_i as usize] = match future {
                        Ok(value) => Some(Arc::new(value)),
                        Err(FlushError::OutOfDate) => {
                            recreate_swapchain = true;
                            None
                        }
                        Err(e) => {
                            println!("failed to flush future: {e}");
                            None
                        }
                    };

                    previous_fence_i = image_i;
                }
                _ => (),
            }
        });
    }

    fn new() -> Self {
        DMapRender {  }
    }
}
