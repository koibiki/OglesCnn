package com.example.cnnlib.layer;

import android.content.Context;

import com.example.cnnlib.render.ComputeRender;
import com.example.cnnlib.utils.AttachIDManager;

import static com.example.cnnlib.render.ComputeRender.getCompShaderLocalSizeY;
import static com.example.cnnlib.render.ComputeRender.initCompPro;

public class NonLinearLayer extends Layer {

    private Layer mPreLayer;
    private NonLinearType mType;
    private int mNumGroupsY;
    private int mShaderPro;

    public enum NonLinearType {
        RELU(0), SIGMOID(1), TANH(2), NONE(-1);

        public int index;
        private NonLinearType(int index) {
            this.index = index;
        }

    }


    public NonLinearLayer(Context context, Layer preLayer, NonLinearType type) {
        super(context, preLayer.getOutputShape());
        this.mPreLayer = preLayer;
        this.mType = type;
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
        mShaderPro = initCompPro(mContext, csPath, mOutputShape[0], localSizeY);
        mAttachID = AttachIDManager.getInstance().getAttachID();
        mOutTex = ComputeRender.createTexture(mAttachID);
    }

    @Override
    public void initialize() {
        initNonlinear();
    }

    @Override
    protected void bindTextureAndBuffer() {
        ComputeRender.bindTextureAndBuffer(mOutTex, mAttachID);
    }

    @Override
    protected void actualForwardProc() {
        ComputeRender.performWithoutParams(mShaderPro, mPreLayer.getOutTex(), mOutTex, mNumGroupsY);
    }

}
