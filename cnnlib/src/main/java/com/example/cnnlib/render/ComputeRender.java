package com.example.cnnlib.render;

import android.content.Context;
import android.opengl.GLES31;

import com.example.cnnlib.model.Kennel;
import com.example.cnnlib.utils.Constants;
import com.example.cnnlib.utils.MathUtils;
import com.example.cnnlib.utils.ShaderUtils;

import java.nio.Buffer;
import java.util.Locale;

import static android.opengl.GLES20.GL_FRAMEBUFFER;
import static android.opengl.GLES20.glReadPixels;
import static android.opengl.GLES20.glUniform1iv;
import static android.opengl.GLES30.glReadBuffer;
import static android.opengl.GLES31.GL_CLAMP_TO_EDGE;
import static android.opengl.GLES31.GL_COLOR_ATTACHMENT0;
import static android.opengl.GLES31.GL_COMPUTE_SHADER;
import static android.opengl.GLES31.GL_FLOAT;
import static android.opengl.GLES31.GL_LINEAR;
import static android.opengl.GLES31.GL_READ_ONLY;
import static android.opengl.GLES31.GL_RGBA;
import static android.opengl.GLES31.GL_RGBA32F;
import static android.opengl.GLES31.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT;
import static android.opengl.GLES31.GL_TEXTURE_2D;
import static android.opengl.GLES31.GL_TEXTURE_MAG_FILTER;
import static android.opengl.GLES31.GL_TEXTURE_MIN_FILTER;
import static android.opengl.GLES31.GL_TEXTURE_WRAP_S;
import static android.opengl.GLES31.GL_TEXTURE_WRAP_T;
import static android.opengl.GLES31.GL_WRITE_ONLY;
import static android.opengl.GLES31.glBindImageTexture;
import static android.opengl.GLES31.glBindTexture;
import static android.opengl.GLES31.glDispatchCompute;
import static android.opengl.GLES31.glFramebufferTexture2D;
import static android.opengl.GLES31.glGenTextures;
import static android.opengl.GLES31.glGetUniformLocation;
import static android.opengl.GLES31.glLinkProgram;
import static android.opengl.GLES31.glMemoryBarrier;
import static android.opengl.GLES31.glTexParameteri;
import static android.opengl.GLES31.glTexStorage2D;
import static android.opengl.GLES31.glTexSubImage2D;
import static android.opengl.GLES31.glUniform1fv;
import static android.opengl.GLES31.glUseProgram;
import static com.example.cnnlib.utils.Constants.S_CONV_SHADER_HEADER;
import static com.example.cnnlib.utils.Constants.S_COMMON_SHADER_HEADER;
import static com.example.cnnlib.utils.Constants.S_POOLING_SHADER_HEADER;
import static com.example.cnnlib.utils.Constants.S_TEXTURE_SIZE;

public class ComputeRender {

    private static final String TAG = "ComputeRender";

    public static int initConvolutePro(Context context, String csPath, Kennel kennel, int xSize, int ySize) {
        int compProg = GLES31.glCreateProgram();
        String source = ShaderUtils.loadFromAssetsFile(csPath, context.getResources());
        source = String.format(Locale.getDefault(), S_CONV_SHADER_HEADER, kennel.area, kennel.size, xSize, ySize) + source;
        ShaderUtils.vglAttachShaderSource(compProg, GL_COMPUTE_SHADER, source);
        glLinkProgram(compProg);
        return compProg;
    }

    public static int initPoolingPro(Context context, String csPath, int pooling_area, int xSize, int ySize) {
        int compProg = GLES31.glCreateProgram();
        String source = ShaderUtils.loadFromAssetsFile(csPath, context.getResources());
        source = String.format(Locale.getDefault(), S_POOLING_SHADER_HEADER, pooling_area, xSize, ySize) + source;
        ShaderUtils.vglAttachShaderSource(compProg, GL_COMPUTE_SHADER, source);
        glLinkProgram(compProg);
        return compProg;
    }

    public static int initCompPro(Context context, String csPath, int xSize, int ySize) {
        int compProg = GLES31.glCreateProgram();
        String source = ShaderUtils.loadFromAssetsFile(csPath, context.getResources());
        source = String.format(Locale.getDefault(), S_COMMON_SHADER_HEADER, xSize, ySize) + source;
        ShaderUtils.vglAttachShaderSource(compProg, GL_COMPUTE_SHADER, source);
        glLinkProgram(compProg);
        return compProg;
    }

    public static int createTexture(int attachID) {
        int[] texture = new int[1];
        glGenTextures(1, texture, 0);
        glBindTexture(GL_TEXTURE_2D, texture[0]);
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, S_TEXTURE_SIZE, S_TEXTURE_SIZE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        GLES31.glBindTexture(GL_TEXTURE_2D, 0);
        int attachment = GL_COLOR_ATTACHMENT0 + attachID;
        glFramebufferTexture2D(GL_FRAMEBUFFER, attachment, GL_TEXTURE_2D, texture[0], 0);
        return texture[0];
    }

    public static int createTexture(int attachID, int width, int height) {
        int[] texture = new int[1];
        glGenTextures(1, texture, 0);
        glBindTexture(GL_TEXTURE_2D, texture[0]);
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, width, height);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        GLES31.glBindTexture(GL_TEXTURE_2D, 0);
        int attachment = GL_COLOR_ATTACHMENT0 + attachID;
        glFramebufferTexture2D(GL_FRAMEBUFFER, attachment, GL_TEXTURE_2D, texture[0], 0);
        return texture[0];
    }

    public static void cleanTexture(int texID) {
        glBindTexture(GL_TEXTURE_2D, texID);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 1024, 1024, GL_RGBA, GL_FLOAT, null);
    }

    public static void performConvolute(int compProg, Kennel kennel, int[] params, int inTex, int outTex, int numGroupsY) {
        glUseProgram(compProg);
        glUniform1fv(glGetUniformLocation(compProg, "k"), kennel.data.length, kennel.data, 0);
        glUniform1iv(glGetUniformLocation(compProg, "params"), params.length, params, 0);

        glBindImageTexture(0, inTex, 0, false, 0, GL_READ_ONLY, GL_RGBA32F);
        glBindImageTexture(1, outTex, 0, false, 0, GL_READ_ONLY, GL_RGBA32F);
        glBindImageTexture(2, outTex, 0, false, 0, GL_WRITE_ONLY, GL_RGBA32F);

        glDispatchCompute(1, numGroupsY, 1);
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    }

    public static void performWithoutParams(int compProg, int inTex, int outTex, int numGroupsY) {
        glUseProgram(compProg);

        glBindImageTexture(0, inTex, 0, false, 0, GL_READ_ONLY, GL_RGBA32F);
        glBindImageTexture(1, outTex, 0, false, 0, GL_WRITE_ONLY, GL_RGBA32F);

        glDispatchCompute(1, numGroupsY, 1);
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    }

    public static void performFullConnect(int compProg, int[] params, int inTex, int outTex, int kennelTex, int numGroupsY) {
        glUseProgram(compProg);

        glUniform1iv(glGetUniformLocation(compProg, "params"), params.length, params, 0);

        glBindImageTexture(0, inTex, 0, false, 0, GL_READ_ONLY, GL_RGBA32F);
        glBindImageTexture(1, kennelTex, 0, false, 0, GL_READ_ONLY, GL_RGBA32F);
        glBindImageTexture(2, outTex, 0, false, 0, GL_WRITE_ONLY, GL_RGBA32F);

        glDispatchCompute(1, numGroupsY, 1);
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    }

    public static void performWithIntParams(int compProg, int[] params, int inputTexture, int outputTexture, int numGroupsY) {
        glUseProgram(compProg);

        glUniform1iv(glGetUniformLocation(compProg, "params"), params.length, params, 0);

        glBindImageTexture(0, inputTexture, 0, false, 0, GL_READ_ONLY, GL_RGBA32F);
        glBindImageTexture(1, outputTexture, 0, false, 0, GL_WRITE_ONLY, GL_RGBA32F);

        glDispatchCompute(1, numGroupsY, 1);
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    }

    public static int getCompShaderLocalSizeY(int[] outputShape) {
        int maxLoaclSizeY = Constants.S_MAX_COMPUTE_SIZE / outputShape[0];
        if (maxLoaclSizeY > outputShape[1]) {
            return outputShape[1];
        } else {
            return (int) Math.pow(2, MathUtils.getPowerBy2(maxLoaclSizeY));
        }
    }

    public static Buffer transferFromTexture(Buffer data, int attachID, int x, int y, int width, int height) {
        glReadBuffer(GL_COLOR_ATTACHMENT0 + attachID);
        glReadPixels(x, y, width, height, GL_RGBA, GL_FLOAT, data);
        return data;
    }

    public static void transferToTexture(Buffer data, int texID, int xOffset, int yOffset, int width, int height) {
        glBindTexture(GL_TEXTURE_2D, texID);
        glTexSubImage2D(GL_TEXTURE_2D, 0, xOffset, yOffset, width, height, GL_RGBA, GL_FLOAT, data);
    }

}
