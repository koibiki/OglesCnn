package com.example.cnnlib.layer;

import android.content.Context;

import com.example.cnnlib.model.LayerParams;
import com.example.cnnlib.render.ComputeRender;
import com.example.cnnlib.utils.AttachIDManager;

import static com.example.cnnlib.render.ComputeRender.getCompShaderLocalSizeY;
import static com.example.cnnlib.render.ComputeRender.initCompPro;

public class NonLinearLayer extends Layer {

    private NonLinearType mType;
    private int mNumGroupsY;
    private int mShaderPro;

    public enum NonLinearType {
        RELU, SIGMOID, TANH
    }


    public NonLinearLayer(Context context, LayerParams layerParams, NonLinearType type) {
        super(context, layerParams);
        this.mType = type;

        initNonlinear();
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
        int localSizeY = getCompShaderLocalSizeY(mLayerParams.outputShape);
        mNumGroupsY = (int) Math.ceil(mLayerParams.outputShape[1] * 1.0d / localSizeY);
        mShaderPro = initCompPro(mContext, csPath, mLayerParams.outputShape[0], localSizeY);
        mAttachID = AttachIDManager.getInstance().getAttachID();
        mOutputTexID = ComputeRender.createTexture(mAttachID);
    }

    @Override
    public int forwardProc(int inputTexID) {
        ComputeRender.performWithoutParams(mShaderPro, inputTexID, mOutputTexID, mNumGroupsY);
        return mOutputTexID;
    }

    @Override
    public void readOutput() {

    }
}
