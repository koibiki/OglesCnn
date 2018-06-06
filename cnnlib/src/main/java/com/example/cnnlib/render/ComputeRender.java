package com.example.cnnlib.render;

import android.content.Context;
import android.opengl.GLES31;
import android.view.ViewGroup;

import com.example.cnnlib.model.Kennel;
import com.example.cnnlib.model.LayerParams;
import com.example.cnnlib.utils.Constants;
import com.example.cnnlib.utils.MathUtils;
import com.example.cnnlib.utils.ShaderUtils;

import java.nio.Buffer;
import java.util.Locale;

import static android.opengl.GLES20.GL_FRAMEBUFFER;
import static android.opengl.GLES20.glReadPixels;
import static android.opengl.GLES20.glUniform1i;
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
import static com.example.cnnlib.utils.Constants.S_TEXTURE_SIZE;

public class ComputeRender {

    private static final String S_COMP_SHADER_HEADER = "#version 310 es\n#define KENNEL_AREA %d\n#define KENNEL_SIZE %d\n#define X_SIZE %d\n#define Y_SIZE %d\n";

    private static int initGLSL(Context context, String csPath, Kennel kennel, int xSize, int ySize) {
        int compProg = GLES31.glCreateProgram();
        String source = ShaderUtils.loadFromAssetsFile(csPath, context.getResources());
        source = String.format(Locale.getDefault(), S_COMP_SHADER_HEADER, kennel.getArea(), kennel.getSize(), xSize, ySize) + source;
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

    public static void performConvolute(Context context, String csPath, int inputTexture, int outputTexture, Kennel kennel, int kennelIndex, LayerParams layerParams, int padding, int paddingType, int[] strides) {
        int localSizeY = getCompShaderLocalSizeY(layerParams.outputShape);
        int compProg = initGLSL(context, csPath, kennel, layerParams.outputShape[0], localSizeY);
        glUseProgram(compProg);

        glUniform1fv(glGetUniformLocation(compProg, "k"), kennel.data.length, kennel.data, 0);
        glUniform1i(glGetUniformLocation(compProg, "kennel_index"), kennelIndex);
        glUniform1i(glGetUniformLocation(compProg, "padding"), padding);
        glUniform1i(glGetUniformLocation(compProg, "padding_type"), paddingType);
        glUniform1iv(glGetUniformLocation(compProg, "strides"), strides.length, strides, 0);
        glUniform1iv(glGetUniformLocation(compProg, "kennel_shape"), kennel.shape.length, kennel.shape, 0);
        glUniform1iv(glGetUniformLocation(compProg, "input_shape"), layerParams.inputShape.length, layerParams.inputShape, 0);
        glUniform1iv(glGetUniformLocation(compProg, "output_shape"), layerParams.outputShape.length, layerParams.outputShape, 0);

        glBindImageTexture(0, inputTexture, 0, false, 0, GL_READ_ONLY, GL_RGBA32F);
        glBindImageTexture(1, outputTexture, 0, false, 0, GL_READ_ONLY, GL_RGBA32F);
        glBindImageTexture(2, outputTexture, 0, false, 0, GL_WRITE_ONLY, GL_RGBA32F);

        int numGroupsY = (int) Math.ceil(layerParams.outputShape[1] * 1.0d / localSizeY);
        glDispatchCompute(1, numGroupsY, 1);
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    }

    private static int getCompShaderLocalSizeY(int[] outputShape) {
        int maxLoaclSizeY = Constants.S_TEXTURE_SIZE / outputShape[0];
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
