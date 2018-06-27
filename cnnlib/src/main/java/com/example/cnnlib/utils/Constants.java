package com.example.cnnlib.utils;

public class Constants {

    public static final String S_CONV_SHADER_HEADER = "#version 310 es\n#define KENNEL_AREA %d\n#define KENNEL_SIZE %d\n#define X_SIZE %d\n#define Y_SIZE %d\n";
    public static final String S_CONV3_SHADER_HEADER = "#version 310 es\n#define KENNEL_AREA %d\n#define KENNEL_AMOUNT %d\n#define KENNEL_SIZE %d\n#define X_SIZE %d\n#define Y_SIZE %d\n#define Z_SIZE %d\n";

    public static final String S_POOLING_SHADER_HEADER = "#version 310 es\n#define KENNEL_AREA %d\n#define X_SIZE %d\n#define Y_SIZE %d\n#define Z_SIZE %d\n";
    public static final String S_FULL_CONN_SHADER_HEADER = "#version 310 es\n#define KENNEL_SIZE %d\n#define KENNEL_AMOUNT %d\n#define X_SIZE %d\n#define Y_SIZE %d\n#define Z_SIZE %d\n";

    public static final String S_COMMON_SHADER_HEADER = "#version 310 es\n#define X_SIZE %d\n#define Y_SIZE %d\n#define Z_SIZE %d\n";

    public static final int S_MAX_COMPUTE_SIZE = 1024;
    public static final int S_TEXTURE_SIZE = S_MAX_COMPUTE_SIZE;
    public static final int S_CONV_KENNEL_TEXTURE_SIZE = 8192;



}
