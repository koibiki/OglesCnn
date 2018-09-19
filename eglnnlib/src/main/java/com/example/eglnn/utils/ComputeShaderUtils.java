package com.example.eglnn.utils;

public class ComputeShaderUtils {

    public static int getCompShaderLocalSizeY(int[] shape) {
        int maxLoaclSizeY = Constants.S_MAX_COMPUTE_SIZE / shape[0];
        if (maxLoaclSizeY > shape[1]) {
            return shape[1];
        } else {
            return (int) Math.pow(2, MathUtils.getPowerBy2(maxLoaclSizeY));
        }
    }

    public static int getCompShaderLocalSizeZ(int[] shape, int count) {
        int maxLoaclSizeZ = Constants.S_MAX_COMPUTE_SIZE / (shape[0] * shape[1]);
        if (maxLoaclSizeZ > shape[2] / count) {
            return shape[2] / count;
        } else {
            return (int) Math.pow(2, MathUtils.getPowerBy2(maxLoaclSizeZ));
        }
    }

}
