package com.example.cnnlib.layer;

import android.content.Context;

import com.example.cnnlib.render.Render;

import static com.example.cnnlib.render.Render.getCompShaderLocalSizeY;
import static com.example.cnnlib.render.Render.initCompPro;

public class NonLinear extends Layer {

    private NonLinearType mType;
    private int mNumGroupsY;
    private int mShaderPro;

    public enum NonLinearType {
        RELU(0), SIGMOID(1), TANH(2), NONE(-1);

        public int index;

        NonLinearType(int index) {
            this.index = index;
        }

    }


    public NonLinear(Context context, Layer preLayer, NonLinearType type) {
        super(context, preLayer);
        this.mType = type;
        this.mOutputShape = preLayer.getOutputShape();
    }

    private void initNonlinear() {
        String csPath = null;
        if (mType == NonLinearType.RELU) {
            csPath = "relu.comp";
        } else if (mType == NonLinearType.SIGMOID) {
            csPath = "sigmoid.comp";
        } else if (mType == NonLinearType.TANH) {
            csPath = "tanh.comp";
        }

        int localSizeY = getCompShaderLocalSizeY(mOutputShape);
        mNumGroupsY = (int) Math.ceil(mOutputShape[1] * 1.0d / localSizeY);
        mShaderPro = initCompPro(mContext, csPath, mOutputShape[0], localSizeY, 1);
        mAttachID = Layer.getDataAttachID();
        mOutTex = Render.createTexture();
    }

    @Override
    public void initialize() {
        initNonlinear();
    }

    @Override
    protected void bindTextureAndBuffer() {
        Render.bindTextureAndBuffer(mOutTex, mAttachID);
    }

    @Override
    protected void actualForwardProc(float[][][] input) {
        Render.performWithoutParams(mShaderPro, mPreLayer.getOutTex(), mOutTex, mNumGroupsY);
    }

}
