import React, { useCallback, useEffect, useMemo, useRef } from "react"

import { Streamlit } from "streamlit-component-lib"
import { useStreamlit } from "streamlit-component-lib-react-hooks"

import { VTKViewer, VTKViewerDrawer, VTKFloatingToolbar } from "lavender-vtkjs"

import { Layout } from "antd"

import './VTKStreamlit.css'

import isequal from "lodash.isequal"
import debounce from "lodash.debounce"

const VTKStreamlit: React.FunctionComponent = () => {
  // "props" aka args coming from Streamlit
  const renderData = useStreamlit()

  // viewerRef contains the reference to the vtkjs viewer api
  const viewerRef = useRef<any>(null)
  // maintain reference to scene state
  const sceneRef = useRef<any[]>()
  const clearRef = useRef<boolean>(true)

  // state returned to streamlit
  const [viewerState, setViewerState] = React.useState<any>({})

  // stack of actions to dispatch via vtkjs
  const actionStackRef = useRef<any[]>([])

  // file to be loaded
  const [file, setFile] = React.useState<Uint8Array | undefined>(undefined)

  // designed to be able to aggreate and dispatch actions on a debounced interval
  // implemented as a standalone function at this point
  const dispatchActionStack = useCallback(() => {
    if (viewerRef.current && viewerRef.current.dispatch &&
      actionStackRef.current && actionStackRef.current.length > 0) {

      // handles screenshot as a special case
      const screenshotIndex = actionStackRef.current.findIndex(a => a.type === "streamlit-screenshot")
      if (screenshotIndex !== -1) handleScreenshot()

      // filters type === "strealit-screenshot", and actions with duplicate types
      // any action with ids [] will be dispatched
      const actions = [...actionStackRef.current].reverse()
        .filter((action, i, array) =>
          (action.type !== "streamlit-screenshot" &&
            typeof action.ids !== 'undefined') ||
          array.findIndex(a => a.type === action.type) === i
        )

      viewerRef.current.dispatch(actions)
      actionStackRef.current = []
    }
  }, [])

  const debouncedDispatch = useCallback(debounce(dispatchActionStack, 250, { maxWait: 750 }), [dispatchActionStack])

  useEffect(() => {
    if (renderData && typeof renderData.args["clear"] !== undefined) {
      clearRef.current = renderData.args["clear"]
    }
  }, [renderData])

  useEffect(() => {
    if (renderData && typeof renderData.args["action_stack"] !== undefined
      && viewerRef.current && viewerRef.current.dispatch) {
      actionStackRef.current = [
        ...actionStackRef.current,
        ...renderData.args["action_stack"]
      ]
      if (actionStackRef.current.length > 0) {
        debouncedDispatch()
      }
    }
  }, [renderData, debouncedDispatch])

  useEffect(() => {
    if (renderData && renderData.args["file"]) {
      setFile(currFile => {
        if (!currFile) return renderData.args["file"]
        const equal = isequal(renderData.args["file"], currFile)
        return equal ? currFile : renderData.args["file"]
      })
    }
  }, [renderData])

  useEffect(() => {
    if (!file) return
    loadFile(file)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [file])

  const loadFile = useCallback((file: Uint8Array) => {
    if (viewerRef.current && viewerRef.current.dispatch && viewerRef.current.loadFile) {
      if (clearRef.current) viewerRef.current.dispatch({ type: 'remove-all' }, true)
      const config = viewerState.scene.length > 0 ? viewerState : undefined
      viewerRef.current.loadFile(new Blob([file]), 'vtkjs', config)
    }
  }, [viewerState])

  // defaults to true
  const toolbar = useMemo(() => {
    if (renderData && typeof renderData.args["toolbar"] !== 'undefined') {
      return renderData.args["toolbar"]
    }
    else {
      return true
    }
  }, [renderData])

  // defaults to true
  const sider = useMemo(() => {
    if (renderData && typeof renderData.args["sider"] !== 'undefined') {
      return renderData.args["sider"]
    }
    else {
      return true
    }
  }, [renderData])

  const cssStyle = useMemo(() => {
    if (renderData && typeof renderData.args["style"] !== 'undefined') {
      if (renderData.args["style"].height && renderData.args["style"].height.includes('px')) {
        Streamlit.setFrameHeight(parseInt(renderData.args["style"].height.replace('px', '')))
      }
      return renderData.args["style"]
    }
    else {
      return { border: "1px solid #d0d7de", borderRadius: "2px" }
    }
  }, [renderData])

  // initial state of streamlit component
  useEffect(() => {
    Streamlit.setComponentValue({})
  }, [])

  useEffect(() => {
    // update if we're subscribed
    // this is camera and everything else
    if (renderData && renderData.args["subscribe"]) {
      Streamlit.setComponentValue(viewerState)
    } else if (viewerState.scene && viewerState.scene.length > 0) {
      const scene = [...viewerState.scene]

      if (isequal(scene, sceneRef.current)) return
      sceneRef.current = scene

      Streamlit.setComponentValue({
        scene
      })
    }
  }, [viewerState, renderData])

  const handleScreenshot = () => {
    if (!viewerRef.current) return
    viewerRef.current.handleScreenshot('VTKJSStreamlit', false)
  }

  if (renderData == null) {
    return (
      <div style={{
        height: "400px",
        border: "1px solid #d0d7de",
        borderRadius: "2px",
        backgroundColor: "#f0f2f5",
      }} />
    )
  }

  return (
    <div style={{ width: '100%', height: '100%', border: "1px solid #d0d7de", borderRadius: "2px", ...cssStyle, display: 'flex' }}>
      <Layout style={{ flexDirection: 'row' }}>
        {sider &&
          <VTKViewerDrawer dispatch={viewerRef.current?.dispatch} viewerState={viewerState} handleScreenshot={handleScreenshot} />
        }
        <Layout>
          {toolbar &&
            <VTKFloatingToolbar dispatch={viewerRef.current?.dispatch} viewerState={viewerState} handleScreenshot={handleScreenshot} />
          }
          <Layout.Content style={{ display: 'flex', flexDirection: 'column' }}>
            <VTKViewer setViewerState={setViewerState} ref={viewerRef} />
          </Layout.Content>
        </Layout>
      </Layout>
    </div>
  )
}

export default VTKStreamlit