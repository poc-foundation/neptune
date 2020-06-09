use std::ptr;
use triton::bindings;
use triton::FutharkContext;

const MAX_LEN: usize = 128;

#[derive(Debug, Clone)]
pub enum ClError {
    DeviceNotFound,
    PlatformNotFound,
}
pub type ClResult<T> = std::result::Result<T, ClError>;

fn get_device_by_bus_id(id: u32) -> ClResult<bindings::cl_device_id> {
    unsafe {
        let mut plats = [ptr::null_mut(); MAX_LEN];
        let mut num_plats = 0u32;
        let res = bindings::clGetPlatformIDs(MAX_LEN as u32, plats.as_mut_ptr(), &mut num_plats);
        assert_eq!(res, bindings::CL_SUCCESS as i32);

        for plat in plats[..num_plats as usize].iter() {
            let mut devs = [ptr::null_mut(); MAX_LEN];
            let mut num_devs = 0u32;
            let res = bindings::clGetDeviceIDs(
                *plat,
                bindings::CL_DEVICE_TYPE_GPU as u64,
                MAX_LEN as u32,
                devs.as_mut_ptr(),
                &mut num_devs,
            );
            assert_eq!(res, bindings::CL_SUCCESS as i32);

            for dev in devs[..num_devs as usize].iter() {
                let mut ret = [0u8; MAX_LEN];
                let mut len = 0u64;
                let res = bindings::clGetDeviceInfo(
                    *dev,
                    0x4008 as u32,
                    MAX_LEN as u64,
                    ret.as_mut_ptr() as *mut std::ffi::c_void,
                    &mut len,
                );
                assert_eq!(res, bindings::CL_SUCCESS as i32);
                assert_eq!(len, 4);
                let bus_id = to_u32(&ret[..4]);
                if bus_id == id {
                    return Ok(*dev);
                }
            }
        }
    }

    Err(ClError::DeviceNotFound)
}

pub fn get_context_by_bus_id(id: u32) -> ClResult<FutharkContext> {
    unsafe {
        let dev = get_device_by_bus_id(id)?;
        let mut res = 0i32;

        let context = bindings::clCreateContext(
            ptr::null(),
            1,
            [dev].as_mut_ptr(),
            None,
            ptr::null_mut(),
            &mut res,
        );
        assert_eq!(res, bindings::CL_SUCCESS as i32);

        let queue = bindings::clCreateCommandQueue(context, dev, 0, &mut res);
        assert_eq!(res, bindings::CL_SUCCESS as i32);

        let ctx_config = bindings::futhark_context_config_new();
        let ctx = bindings::futhark_context_new_with_command_queue(ctx_config, queue);
        Ok(FutharkContext {
            context: ctx,
            config: ctx_config,
        })
    }
}

fn to_u32(inp: &[u8]) -> u32 {
    (inp[0] as u32) + ((inp[1] as u32) << 8) + ((inp[2] as u32) << 16) + ((inp[3] as u32) << 24)
}
