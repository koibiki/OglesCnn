package com.example.cnnlib.layer;

import android.content.Context;

import com.example.cnnlib.render.ComputeRender;

import static com.example.cnnlib.render.ComputeRender.getCompShaderLocalSizeY;
import static com.example.cnnlib.render.ComputeRender.getCompShaderLocalSizeZ;
import static com.example.cnnlib.render.ComputeRender.initCompPro;

/**
 * 输出维度的channel 需要 4 对齐
 * 存储方式为 x 为 上层输出的channel / 4
 * y 为 上层输出的 width * height
 * <p>
 * 在shader中, 计算器坐标 与输出纹理坐标对应关系
 * t.x = pos.z
 * t.y = pos.y * inputShape[0] + pos.x
 * 计算器坐标 与输入纹理坐标对应关系
 * t.x = pos.x
 * t.y = pos.y
 * t.index = pos.z
 */
public class FlatLayer extends Layer {

    private int mShaderPro;
    private int[] mParams;
    private int mNumGroupsY;
    private int mNumGroupsZ;

    public FlatLayer(Context context, Layer preLayer) {
        super(context, preLayer);
    }

    private void initFlat() {
        int[] inputShape = mPreLayer.getOutputShape();
        int localSizeY = getCompShaderLocalSizeY(inputShape);
        mNumGroupsY = (int) Math.ceil(inputShape[1] * 1.0d / localSizeY);
        int localSizeZ = getCompShaderLocalSizeZ(inputShape, 4);
        mNumGroupsZ = (int) Math.ceil(inputShape[2] * 1.0d / (localSizeZ + 4));


        mShaderPro = initCompPro(mContext, "flat.comp", inputShape[0], localSizeY, localSizeZ);
        mAttachID = Layer.getDataAttachID();
        mOutTex = ComputeRender.createTexture();

        mParams = new int[10];
        mParams[0] = inputShape[0];
        mParams[1] = inputShape[1];
        mParams[2] = inputShape[2];
        mParams[3] = mOutputShape[0];
        mParams[4] = mOutputShape[1];
        mParams[5] = mOutputShape[2];
    }

    @Override
    public void initialize() {
        initFlat();
    }


    @Override
    protected void bindTextureAndBuffer() {
        ComputeRender.bindTextureAndBuffer(mOutTex, mAttachID);
    }

    @Override
    protected void actualForwardProc() {
        ComputeRender.performWithIntParams(mShaderPro, mParams, mPreLayer.getOutTex(), mOutTex, mNumGroupsY, mNumGroupsZ);
    }

}
