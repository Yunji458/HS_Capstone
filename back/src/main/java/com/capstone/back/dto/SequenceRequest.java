package com.capstone.back.dto;

import java.util.List;

public class SequenceRequest {
    private List<List<Double>> sequence;

    public List<List<Double>> getSequence() {
        return sequence;
    }

    public void setSequence(List<List<Double>> sequence) {
        this.sequence = sequence;
    }
}
