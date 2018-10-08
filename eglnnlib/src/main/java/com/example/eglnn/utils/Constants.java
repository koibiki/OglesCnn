package com.example.eglnn.utils;

public class Constants {

    public static final String S_SOFTMAX_SHADER_HEADER =
            "#version 310 es\n" +
                    "#define AMOUNT %d\n" +
                    "#define X_SIZE %d\n" +
                    "#define Y_SIZE %d\n" +
                    "#define Z_SIZE %d\n";

    public static final String S_FULL_CONN_SHADER_HEADER =
            "#version 310 es\n" +
                    "#define KERNEL_SIZE %d\n" +
                    "#define KERNEL_AMOUNT %d\n" +
                    "#define X_SIZE %d\n" +
                    "#define Y_SIZE %d\n" +
                    "#define Z_SIZE %d\n";

    public static final String S_COMMON_SHADER_HEADER =
            "#version 310 es\n" +
                    "#define X_SIZE %d\n" +
                    "#define Y_SIZE %d\n" +
                    "#define Z_SIZE %d\n";

    public static final int S_TEXTURE_SIZE = 1024;

}
