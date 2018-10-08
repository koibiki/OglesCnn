package com.example.eglnn.utils;

import com.example.eglnn.eglenv.GLES31BackEnv;

public class ComputeShaderUtils {

    public static int getCompShaderLocalSizeY(int[] shape) {
        int maxLoaclSizeY = GLES31BackEnv.getMaxWorkGroupSize() / shape[0];
        if (maxLoaclSizeY > shape[1]) {
            return shape[1];
        } else {
            return (int) Math.pow(2, MathUtils.getPowerBy2(maxLoaclSizeY));
        }
    }

    public static int getCompShaderLocalSizeZ(int[] shape, int count) {
        int maxLoaclSizeZ = GLES31BackEnv.getMaxWorkGroupSize() / (shape[0] * shape[1]);
        if (maxLoaclSizeZ > shape[2] / count) {
            return (int) Math.ceil(shape[2] * 1.0f / count);
        } else {
            return (int) Math.pow(2, MathUtils.getPowerBy2(maxLoaclSizeZ));
        }
    }

}
