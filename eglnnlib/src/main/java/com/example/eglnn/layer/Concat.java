package com.example.eglnn.layer;

import android.content.Context;

import com.example.eglnn.Render;
import com.example.eglnn.utils.ShaderUtils;
import com.example.eglnn.utils.Utils;

import java.util.Locale;

import static com.example.eglnn.Render.initCompPro;
import static com.example.eglnn.utils.ComputeShaderUtils.getCompShaderLocalSizeY;
import static com.example.eglnn.utils.ComputeShaderUtils.getCompShaderLocalSizeZ;
import static com.example.eglnn.utils.Constants.S_COMMON_SHADER_HEADER;

/**
 * axis 0 , 1, 2 w, h, c
 */
public class Concat extends Layer {

    private int mAxis;
    private int mNumGroupsY;
    private int mNumGroupsZ;
    private int mShaderPro;
    private int[] mParams;

    public Concat(Context context, String name, Layer[] preLayers, int axis) {
        super(context, name, preLayers);
        this.mAxis = axis;
        this.mOutShape = calculateOutShape();
    }

    private int[] calculateOutShape() {
        int[] widths = new int[mPreLayers.length];
        int[] heights = new int[mPreLayers.length];
        int[] channels = new int[mPreLayers.length];

        for (int i = 0; i < mPreLayers.length; i++) {
            widths[i] = mPreLayers[i].getOutputShape()[0];
            heights[i] = mPreLayers[i].getOutputShape()[1];
            channels[i] = mPreLayers[i].getOutputShape()[2];
        }
        if (mAxis == 0) {
            checkShape(heights);
            checkShape(channels);
        } else if (mAxis == 1) {
            checkShape(widths);
            checkShape(channels);
        } else {
            checkShape(widths);
            checkShape(heights);
        }

        if (mAxis == 0) {
            return new int[]{sum(widths), heights[0], channels[0]};
        } else if (mAxis == 1) {
            return new int[]{widths[0], sum(heights), channels[0]};
        } else {
            return new int[]{widths[0], heights[0], sum(channels)};
        }
    }


    private int sum(int[] shapes) {
        int sum = 0;
        for (int shape : shapes) {
            sum += shape;
        }
        return sum;
    }

    private void checkShape(int[] shapes) {
        int bench = shapes[0];
        for (int shape : shapes) {
            if (bench != shape) {
                try {
                    throw new Exception("维度不匹配");
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }
    }


    @Override
    public void initialize() {
        initConcat();
    }

    private void initConcat() {
        int localSizeY = getCompShaderLocalSizeY(mOutShape);
        mNumGroupsY = (int) Math.ceil(mOutShape[1] * 1.0d / localSizeY);
        int localSizeZ = getCompShaderLocalSizeZ(mOutShape, 4);
        mNumGroupsZ = (int) Math.ceil(mOutShape[2] * 1.0d / (localSizeZ * 4));

        String source = createShaderSource(localSizeY, localSizeZ);

        mShaderPro = initCompPro(source);
        mAttachID = Layer.getDataAttachID();
        mOutTex = Render.createFloatTextureArray(mOutShape[0], mOutShape[1], Utils.alignBy4(mOutShape[2]) / 4);

        // TODO 先用ubo 以后换为 SSBO 动态传入shader中
        mParams = new int[13];
        mParams[0] = mPreLayers[0].getOutputShape()[0];
        mParams[1] = mPreLayers[0].getOutputShape()[1];
        mParams[2] = Utils.alignBy4(mPreLayers[0].getOutputShape()[2]) / 4;
        mParams[3] = mPreLayers[1].getOutputShape()[0];
        mParams[4] = mPreLayers[1].getOutputShape()[1];
        mParams[5] = Utils.alignBy4(mPreLayers[1].getOutputShape()[2]) / 4;
        mParams[6] = mOutShape[0];
        mParams[7] = mOutShape[1];
        mParams[8] = mOutShape[2];
        mParams[9] = Utils.alignBy4(mOutShape[2]) / 4;
    }

    private String createShaderSource(int localSizeY, int localSizeZ) {
        String shaderFile = "concat_axis" + mAxis + ".comp";
        String source = ShaderUtils.loadFromAssetsFile(shaderFile, mContext.getResources());
        return String.format(Locale.getDefault(), S_COMMON_SHADER_HEADER, mOutShape[0], localSizeY, localSizeZ) + source;
    }

    @Override
    protected void bindTextureAndBuffer() {
        Render.bindTextureArray(mOutTex, mAttachID);
    }

    @Override
    protected void actualForwardProc(float[][] input) {
        Render.performConcat(mShaderPro, mParams, mPreLayers[0].getOutTex(), mPreLayers[1].getOutTex(), mOutTex, 1, mNumGroupsY, mNumGroupsZ);
    }
}
