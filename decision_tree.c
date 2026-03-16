const char* select_impl(int num_frames, int width, int height, int total_pixels, int pixels_per_frame, int ranks, int threads, int nodes, int cuda_available) {
    if (cuda_available <= 0.5) {
        if (threads <= 3.0) {
            if (width <= 115.0) {
                return "SEQ";
            } else {
                if (nodes <= 1.5) {
                    return "MPI";
                } else {
                    if (ranks <= 3.0) {
                        return "MPI";
                    } else {
                        return "OMP";
                    }
                }
            }
        } else {
            if (total_pixels <= 61700.0) {
                if (width <= 115.0) {
                    return "SEQ";
                } else {
                    if (threads <= 6.0) {
                        return "OMP";
                    } else {
                        return "SEQ";
                    }
                }
            } else {
                return "OMP";
            }
        }
    } else {
        if (total_pixels <= 1854500.0) {
            if (pixels_per_frame <= 20900.0) {
                return "SEQ";
            } else {
                if (threads <= 3.0) {
                    if (total_pixels <= 491625.0) {
                        return "MPI";
                    } else {
                        return "CUDA";
                    }
                } else {
                    if (total_pixels <= 491625.0) {
                        return "OMP";
                    } else {
                        return "OMP";
                    }
                }
            }
        } else {
            if (total_pixels <= 8100000.0) {
                if (height <= 490.0) {
                    return "CUDA";
                } else {
                    if (threads <= 3.0) {
                        return "CUDA";
                    } else {
                        return "OMP";
                    }
                }
            } else {
                return "CUDA";
            }
        }
    }
}
