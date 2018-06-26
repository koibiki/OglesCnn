package com.example.cnnlib.layer;

import android.content.Context;

import com.example.cnnlib.render.ComputeRender;
import com.example.cnnlib.utils.AttachIDManager;
import com.example.cnnlib.utils.DataUtils;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;

import static com.example.cnnlib.render.ComputeRender.getCompShaderLocalSizeY;
import static com.example.cnnlib.render.ComputeRender.initConvolutePro;

/**
 * 使用纹理(fbo) 存储kennel, 每行存储一个kennel, 每个值的channel按照4对齐, 然后依次排列每个值, 每个值的通道排完后,
 * 再排列下一个值
 * <p>
 * TODO 在shader中, 为避免多线程冲突, 每个计算器同时计算输出的4个通道
 */
public class ConvolutionLayer2 extends Layer {

    private static final String TAG = "ConvolutionLayer";

    private List<float[]> mKennels;
    private int[] mStrides;
    private int[] mKennelShape;
    private int mPadding;
    private int mShaderPro;
    private int mNumGroupsY;
    private int[] mParams;
    private NonLinearLayer.NonLinearType mType;

    private int mKennelTex;
    private int mKennelAttachId;


    public ConvolutionLayer2(Context context, Layer preLayer, int[] shape, int[] kennelShape, int padding, int[] strides, NonLinearLayer.NonLinearType type) {
        super(context, shape, preLayer);
        this.mKennelShape = kennelShape;
        this.mPadding = padding;
        this.mStrides = strides;
        this.mType = type;
    }

    private void initConv() {
        int localSizeY = getCompShaderLocalSizeY(mOutputShape);
        mNumGroupsY = (int) Math.ceil(mOutputShape[1] * 1.0d / localSizeY);
        mShaderPro = initConvolutePro(mContext, "conv2.comp", mKennelShape, mOutputShape[0], localSizeY);
        mAttachID = AttachIDManager.getInstance().getDataAttachID();
        mOutTex = ComputeRender.createTexture();

        mKennels = createKennels();

        mKennelAttachId = AttachIDManager.getInstance().getConvAttachId();
        mKennelTex = ComputeRender.createConvKennelTexture();

        int[] kennelRestoreStartIndex = AttachIDManager.getInstance().getLastConvKennelIndex();
        transferKennelToTex(mKennelTex, kennelRestoreStartIndex[3]);

        int startY = kennelRestoreStartIndex[3];
        int endY = kennelRestoreStartIndex[3] + mOutputShape[2];
        AttachIDManager.getInstance().saveConvKennelIndex(new int[]{0, startY, 0, endY});

        mParams = new int[17];
        int[] inputShape = mPreLayer.getOutputShape();
        mParams[0] = mKennelShape[0];
        mParams[1] = mKennelShape[1];
        mParams[2] = mKennelShape[2];
        mParams[3] = inputShape[0];
        mParams[4] = inputShape[1];
        mParams[5] = inputShape[2];
        mParams[6] = mOutputShape[0];
        mParams[7] = mOutputShape[1];
        mParams[8] = mOutputShape[2];
        mParams[9] = mStrides[0];
        mParams[10] = mStrides[1];
        mParams[11] = mPadding;
        mParams[12] = mType.index;
        mParams[13] = kennelRestoreStartIndex[3];       // 起始点的y坐标
    }

    private List<float[]> createKennels() {
        List<float[]> kennels = new ArrayList<>();
        for (int i = 0; i < mOutputShape[2]; i++) {
            kennels.add(DataUtils.createConvKennel2(i + 1, 3, 3, 4, 1));
        }
        return kennels;
    }

    // kennel size 不能大于 4 * 8192
    private void transferKennelToTex(int tex, int yOffset) {
        for (int i = 0; i < mKennels.size(); i++) {
            ComputeRender.transferToTexture(FloatBuffer.wrap(mKennels.get(i)), tex, 0, yOffset + i, mKennels.get(i).length / 4, 1);
        }
    }

    @Override
    public void initialize() {
        initConv();
    }

    @Override
    protected void bindTextureAndBuffer() {
        ComputeRender.bindTextureAndBuffer(mKennelTex, mKennelAttachId);
        ComputeRender.bindTextureAndBuffer(mOutTex, mAttachID);
    }

    @Override
    protected void actualForwardProc() {
        ComputeRender.performConvolute(mShaderPro, mParams, mPreLayer.getOutTex(), mOutTex, mKennelTex, mNumGroupsY, mOutputShape[2]);
    }

}
