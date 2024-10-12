<script setup lang="ts">
import { ref } from 'vue'
import { invoke } from '@tauri-apps/api/core';

const matrix1 = ref("")
const matrix2 = ref("")
const matrix3 = ref("")
async function summ() {
  await invoke("summ_matrix", {matrix_string1: matrix1.value, matrix_string2: matrix2.value})
    .then((message) => matrix3.value = message)
    .catch((message) => matrix3.value = message)
}

async function div() {
  await invoke("div_matrix", {matrix_string1: matrix1.value, matrix_string2: matrix2.value})
    .then((message) => matrix3.value = message)
    .catch((message) => matrix3.value = message)
}

async function mul() {
  await invoke("mul_matrix", {matrix_string1: matrix1.value, matrix_string2: matrix2.value})
    .then((message) => matrix3.value = message)
    .catch((message) => matrix3.value = message)
}
</script>

<template>
  <div class="flex flex-row h-3/5">
    <div class="w-2/5 h-full">
        <div class="w-full bg-slate-700 h-2/3">
          <textarea
            class="bg-transparent w-full h-full resize-none p-4"
            v-model="matrix1"
          ></textarea>
        </div>
        <div>

        </div>
    </div>
    <div class="flex flex-col w-1/5 h-2/3 px-4 justify-end">
        <button @click="summ">
          <div class="mx-auto bg-slate-700 text-center p-2 w-3/5">A+B</div>
        </button>
        <button @click="div">
          <div class="mx-auto bg-slate-700 text-center p-2 w-3/5 mt-4">A-B</div>
        </button>
        <button @click="mul">
          <div class="mx-auto bg-slate-700 text-center p-2 w-3/5 mt-4">A*B</div>
        </button>
    </div>
    <div class="w-2/5 h-full">
        <div class="w-full bg-slate-700 h-2/3">
          <textarea
            class="bg-transparent w-full h-full resize-none p-4"
            v-model="matrix2"
          ></textarea>
        </div>
        <div>

        </div>
    </div>
  </div>
  <div class="h-2/5 w-full bg-red">
    <div class="bg-slate-700 w-2/5 h-full">
          <textarea
            class="bg-transparent w-full h-full resize-none p-4"
            v-model="matrix3"
          ></textarea>
        </div>
  </div>
  <div>

  </div>
</template>